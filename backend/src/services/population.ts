// [NOTE] There is no backend logic here related to graph size; this must be handled on the frontend in e.g. PopulationTable.tsx or CSS.
// Keeping this file unchanged as the backend has no effect on "graph is too small".

import axios from "axios";
import AdmZip from "adm-zip";
import type { IZipEntry } from "adm-zip";
import { parse } from "csv-parse/sync";
import { redis } from "../lib/cache.js";
import { withClient } from "../lib/db.js";

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

let fetchQueue: Promise<void> = Promise.resolve();
let lastFetchTime = 0;
const MIN_INTERVAL_MS = 1500;
const worldBankCookies: Record<string, string> = {};
type BulkIndicatorCache = {
  indicatorName: string;
  countries: Map<string, DataRow[]>;
};
const bulkDownloadCache = new Map<string, BulkIndicatorCache>();

function buildCookieHeader(): string | undefined {
  const entries = Object.entries(worldBankCookies);
  if (entries.length === 0) return undefined;
  return entries.map(([key, value]) => `${key}=${value}`).join("; ");
}

function storeCookies(setCookieHeader?: string[] | string) {
  if (!setCookieHeader) return;
  const cookies = Array.isArray(setCookieHeader)
    ? setCookieHeader
    : [setCookieHeader];

  cookies.forEach((cookie) => {
    const [pair] = cookie.split(";");
    if (!pair) return;
    const [key, value] = pair.split("=");
    if (key && value) {
      worldBankCookies[key.trim()] = value.trim();
    }
  });
}

async function scheduleRateLimitedFetch<T>(task: () => Promise<T>): Promise<T> {
  const chained = fetchQueue
    .catch(() => undefined)
    .then(async () => {
      const now = Date.now();
      const waitFor = Math.max(0, MIN_INTERVAL_MS - (now - lastFetchTime));
      if (waitFor > 0) {
        await sleep(waitFor);
      }
      const result = await task();
      lastFetchTime = Date.now();
      return result;
    });

  fetchQueue = chained.then(
    () => undefined,
    () => undefined
  );
  return chained;
}

async function fetchWithRetry(url: string, retries = 3, initialDelayMs = 500) {
  let attempt = 0;
  let delayMs = initialDelayMs;

  while (attempt <= retries) {
    try {
      return await scheduleRateLimitedFetch(() =>
        axios.get(url, {
          headers: {
            "User-Agent":
              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            Accept: "application/json",
            Cookie: buildCookieHeader(),
          },
          timeout: 30000,
        })
      );
    } catch (error: any) {
      const setCookie = error?.response?.headers?.["set-cookie"];
      if (setCookie) {
        storeCookies(setCookie as string[]);
      }

      const status = error?.response?.status;
      const shouldRetry =
        attempt < retries &&
        (status === 429 || status === 503 || status >= 500);

      if (!shouldRetry) {
        throw error;
      }

      if (setCookie) {
        console.info(
          `[getWorldBankData] Stored cookies after status ${status}`
        );
      }

      console.warn(
        `[getWorldBankData] World Bank API returned ${status}. Retrying in ${delayMs}ms (attempt ${
          attempt + 1
        }/${retries})`
      );
      await sleep(delayMs);
      delayMs *= 2; // Exponential backoff
      attempt += 1;
    }
  }

  throw new Error("World Bank API request failed after retries");
}

function sanitizeIndicatorName(raw: any, fallback: string): string {
  if (!raw) return fallback;
  if (typeof raw === "string") return raw;
  if (typeof raw === "object" && raw.value) return String(raw.value);
  return fallback;
}

async function fetchFromBulkDownload(
  indicator: string
): Promise<BulkIndicatorCache> {
  if (bulkDownloadCache.has(indicator)) {
    return bulkDownloadCache.get(indicator)!;
  }

  const bulkUrl = `https://api.worldbank.org/v2/en/indicator/${indicator}?downloadformat=csv`;
  console.warn(
    `[getWorldBankData] Falling back to bulk download for ${indicator}: ${bulkUrl}`
  );

  const response = await scheduleRateLimitedFetch(() =>
    axios.get(bulkUrl, {
      responseType: "arraybuffer",
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        Accept: "application/zip,application/octet-stream",
        Cookie: buildCookieHeader(),
      },
      timeout: 60000,
    })
  );

  const setCookie = response.headers?.["set-cookie"];
  if (setCookie) {
    storeCookies(setCookie as string[]);
  }

  const zip = new AdmZip(Buffer.from(response.data));
  const csvEntry = zip
    .getEntries()
    .find(
      (entry: IZipEntry) =>
        /API_.*\.csv$/i.test(entry.entryName) &&
        entry.entryName.includes(indicator)
    );

  if (!csvEntry) {
    throw new Error(
      `Bulk download did not contain expected CSV for indicator ${indicator}`
    );
  }

  const csvText = csvEntry.getData().toString("utf8");
  const lines: string[] = csvText.split(/\r?\n/);
  const headerIndex = lines.findIndex((line: string) =>
    line.startsWith("Country Name,Country Code,Indicator Name,Indicator Code")
  );

  if (headerIndex === -1) {
    throw new Error(
      `Bulk download CSV for ${indicator} missing expected header row`
    );
  }

  const dataSection = lines.slice(headerIndex).join("\n");
  const records: any[] = parse(dataSection, {
    columns: true,
    skip_empty_lines: true,
  });

  const countries = new Map<string, DataRow[]>();
  let indicatorName = indicator;

  for (const record of records) {
    const countryCode = String(record["Country Code"] || "").toUpperCase();
    if (!countryCode) continue;

    if (indicatorName === indicator && record["Indicator Name"]) {
      indicatorName = sanitizeIndicatorName(
        record["Indicator Name"],
        indicator
      );
    }

    const rows: DataRow[] = [];
    for (const key of Object.keys(record)) {
      if (/^\d{4}$/.test(key)) {
        const year = Number(key);
        const rawValue = record[key];
        if (rawValue !== undefined && rawValue !== null && rawValue !== "") {
          const value = Number(rawValue);
          if (!Number.isNaN(value)) {
            rows.push({
              year,
              value,
              countryCode,
              countryName: String(record["Country Name"] || countryCode),
            });
          }
        }
      }
    }

    if (rows.length > 0) {
      rows.sort((a, b) => a.year - b.year);
      countries.set(countryCode, rows);
    }
  }

  const cacheEntry: BulkIndicatorCache = {
    indicatorName,
    countries,
  };

  bulkDownloadCache.set(indicator, cacheEntry);
  return cacheEntry;
}

export type DataRow = {
  year: number;
  value: number | null;
  countryCode: string;
  countryName: string;
};

/**
 * Generate a cache key using indicator, country, and optional year range.
 */
function cacheKey(
  indicator: string,
  country: string,
  from?: number,
  to?: number
): string {
  return `wb:${indicator}:${country}:${from ?? "min"}:${to ?? "max"}`;
}

/**
 * Fetch data for any World Bank indicator for a country.
 * Returns the rows and *the human-readable indicator label* as `indicatorName`,
 * so the frontend can label axes/graph/title properly and reflect the selected indicator.
 */
export async function getWorldBankData(
  indicator: string,
  country: string,
  from?: number,
  to?: number
): Promise<{ rows: DataRow[]; indicatorName: string }> {
  // Validate indicator is provided
  if (!indicator || indicator.trim() === "") {
    throw new Error("Indicator is required");
  }

  // Try cache first to preserve indicatorName, which enables correct labeling
  const key = cacheKey(indicator, country, from, to);
  console.log(
    `[getWorldBankData] Requesting ${indicator} for ${country}, cache key: ${key}`
  );
  const cached = await redis.get(key);
  if (cached) {
    const parsed = JSON.parse(cached);
    // Validate cached data matches the requested indicator
    if (parsed.indicatorName && parsed.rows && parsed.rows.length > 0) {
      console.log(
        `[getWorldBankData] Cache hit for ${key}, returning ${parsed.rows.length} rows, indicatorName: ${parsed.indicatorName}`
      );
      return { rows: parsed.rows, indicatorName: parsed.indicatorName };
    } else {
      console.log(`[getWorldBankData] Cache data invalid, fetching fresh data`);
    }
  }
  console.log(
    `[getWorldBankData] Cache miss for ${key}, fetching from World Bank API`
  );

  // Build World Bank API query parameter for date filtering
  let dateParam: string | undefined = undefined;
  if (from && to) dateParam = `${from}:${to}`;
  else if (from) dateParam = `${from}`;
  else if (to) dateParam = `${to}`;

  // Build the World Bank Data API URL (NOT the Documents API)
  // This is for time series data: https://api.worldbank.org/v2/country/{country}/indicator/{indicator}
  const url = new URL(
    `https://api.worldbank.org/v2/country/${country}/indicator/${indicator}`
  );
  url.searchParams.set("format", "json");
  if (dateParam) url.searchParams.set("date", dateParam);
  url.searchParams.set("per_page", "10000");

  console.log(`[getWorldBankData] Calling World Bank API: ${url.toString()}`);
  let data;
  try {
    const response = await fetchWithRetry(url.toString(), 5, 800);
    const receivedCookies = response.headers?.["set-cookie"];
    if (receivedCookies) {
      storeCookies(receivedCookies as string[]);
    }
    data = response.data;
    console.log(
      `[getWorldBankData] World Bank API response: data array length=${
        Array.isArray(data) ? data.length : "not array"
      }, data[1] length=${
        Array.isArray(data) && data[1] ? data[1].length : "N/A"
      }`
    );
  } catch (error: any) {
    // Log the error for debugging
    console.error(
      `World Bank API error for ${indicator}/${country}:`,
      error.message
    );
    if (error.response) {
      console.error(`Response status: ${error.response.status}`);
      console.error(`Response data:`, error.response.data);
    }

    try {
      const fallback = await fetchFromBulkDownload(indicator);
      const fallbackRows = fallback.countries.get(country) ?? [];
      const filteredRows = fallbackRows.filter((row) => {
        if (from && row.year < from) return false;
        if (to && row.year > to) return false;
        return true;
      });

      if (filteredRows.length > 0) {
        console.warn(
          `[getWorldBankData] Using bulk download fallback for ${indicator}/${country} with ${fallbackRows.length} rows`
        );
        return { rows: filteredRows, indicatorName: fallback.indicatorName };
      }
      console.warn(
        `[getWorldBankData] Bulk download fallback had no rows for ${indicator}/${country}`
      );
    } catch (bulkError: any) {
      console.error(
        `[getWorldBankData] Bulk download fallback failed for ${indicator}/${country}: ${bulkError?.message}`
      );
    }

    return { rows: [], indicatorName: indicator };
  }

  // Check if we got an error response from World Bank API
  if (!Array.isArray(data) || data.length === 0) {
    console.warn(
      `[getWorldBankData] Invalid response structure for ${indicator}/${country}`
    );
    return { rows: [], indicatorName: indicator };
  }

  // Check if World Bank API returned an error message
  if (data[0] && data[0].message) {
    const errorMsg = Array.isArray(data[0].message)
      ? data[0].message.map((m: any) => m.value || m.key || "").join(", ")
      : String(data[0].message);
    console.warn(
      `[getWorldBankData] World Bank API error for ${indicator}/${country}: ${errorMsg}`
    );
    return { rows: [], indicatorName: indicator };
  }

  // World Bank API returns [metadata, dataArray] format
  // Check if we have the data array
  if (data.length < 2 || !Array.isArray(data[1])) {
    // No data available for this indicator/country combination
    // But try to extract indicator name from metadata if available
    let indicatorName: string = indicator;
    if (data[0] && data[0].indicator) {
      if (typeof data[0].indicator === "string") {
        indicatorName = data[0].indicator;
      } else if (data[0].indicator.value) {
        indicatorName = data[0].indicator.value;
      }
    }
    console.warn(
      `[getWorldBankData] No data available for ${indicator}/${country}, but extracted name: ${indicatorName}`
    );
    return { rows: [], indicatorName };
  }

  // Extract indicator label for graph labeling. Fallback to indicator code if not present.
  let indicatorName: string = indicator;

  // Verify the indicator in the response matches what we requested
  if (data[1] && Array.isArray(data[1]) && data[1].length > 0) {
    const firstEntry = data[1][0];
    if (firstEntry && firstEntry.indicator) {
      const responseIndicatorId =
        firstEntry.indicator.id || firstEntry.indicator;
      if (
        responseIndicatorId &&
        responseIndicatorId.toUpperCase() !== indicator.toUpperCase()
      ) {
        console.warn(
          `[getWorldBankData] Indicator mismatch! Requested: ${indicator}, Got: ${responseIndicatorId}`
        );
      }

      // Get indicator name from the response
      if (
        typeof firstEntry.indicator === "object" &&
        firstEntry.indicator.value
      ) {
        indicatorName = firstEntry.indicator.value;
      } else if (typeof firstEntry.indicator === "string") {
        indicatorName = firstEntry.indicator;
      }
    }

    // Fallback: try to find indicator name in any entry
    if (indicatorName === indicator) {
      const nonEmptyEntry = (data[1] as any[]).find(
        (entry: any) =>
          entry &&
          entry.indicator &&
          typeof entry.indicator.value === "string" &&
          entry.indicator.value.trim() !== ""
      );
      if (nonEmptyEntry) {
        indicatorName = nonEmptyEntry.indicator.value;
      }
    }
  }

  // Final fallback to metadata
  if (indicatorName === indicator && data[0] && data[0].indicator) {
    if (typeof data[0].indicator === "string") {
      indicatorName = data[0].indicator;
    } else if (data[0].indicator.value) {
      indicatorName = data[0].indicator.value;
    }
  }

  console.log(
    `[getWorldBankData] Extracted indicatorName: ${indicatorName} for indicator: ${indicator}`
  );

  // Map the data array to our DataRow format
  // Handle both null values and ensure we're getting valid data
  const rows: DataRow[] = (data[1] as any[])
    .filter((d: any) => {
      // Verify each entry matches the requested indicator
      if (d && d.indicator) {
        const entryIndicator =
          typeof d.indicator === "object" ? d.indicator.id : d.indicator;
        if (
          entryIndicator &&
          entryIndicator.toUpperCase() !== indicator.toUpperCase()
        ) {
          console.warn(
            `[getWorldBankData] Entry indicator mismatch! Requested: ${indicator}, Entry has: ${entryIndicator}`
          );
          return false; // Skip entries that don't match
        }
      }
      return d && d.date; // Filter out invalid entries
    })
    .map((d: any) => ({
      year: Number(d.date),
      value: d.value === null || d.value === undefined ? null : Number(d.value),
      countryCode: d.country?.id ?? country,
      countryName: d.country?.value ?? country,
    }))
    .filter((row) => !isNaN(row.year)); // Filter out invalid years

  if (rows.length === 0) {
    console.warn(
      `[getWorldBankData] No valid rows found for ${indicator}/${country} after filtering`
    );
  }

  console.log(
    `[getWorldBankData] Processed ${rows.length} rows for ${indicator}/${country}, indicatorName: ${indicatorName}, sample value: ${rows[0]?.value}`
  );

  await persistRows(indicator, indicatorName, rows);

  // Cache the result, especially the indicatorName, so the graph can be labeled
  const cacheData = { rows, indicatorName };
  await redis.set(key, JSON.stringify(cacheData), "EX", 60 * 60); // 1 hour
  console.log(`[getWorldBankData] Cached data with key: ${key}`);

  return cacheData;
}

/**
 * Store rows (with indicator label) to DB for future analytics/use.
 */
async function persistRows(
  indicatorCode: string,
  indicatorName: string,
  rows: DataRow[]
): Promise<void> {
  if (rows.length === 0) return;
  const countryCode = rows[0].countryCode;
  const countryName = rows[0].countryName;

  await withClient(async (client) => {
    await client.query("BEGIN");
    try {
      const countryRes = await client.query(
        `INSERT INTO countries (iso3_code, name)
         VALUES ($1, $2)
         ON CONFLICT (iso3_code) DO UPDATE SET name = EXCLUDED.name
         RETURNING id`,
        [countryCode, countryName]
      );
      const countryId = countryRes.rows[0].id as number;

      const indRes = await client.query(
        `INSERT INTO indicators (code, name)
         VALUES ($1, $2)
         ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
         RETURNING id`,
        [indicatorCode, indicatorName]
      );
      const indicatorId = indRes.rows[0].id as number;

      for (const r of rows) {
        await client.query(
          `INSERT INTO observations (country_id, indicator_id, year, value)
           VALUES ($1, $2, $3, $4)
           ON CONFLICT (country_id, indicator_id, year) DO UPDATE SET value = EXCLUDED.value`,
          [countryId, indicatorId, r.year, r.value]
        );
      }
      await client.query("COMMIT");
    } catch (e) {
      await client.query("ROLLBACK");
      throw e;
    }
  });
}

/**
 * For compatibility: get population just calls getWorldBankData with correct indicator.
 * Returns only the array of data rows (the indicatorName can be ignored by caller if unused).
 */
export async function getPopulation(
  country: string,
  from?: number,
  to?: number
): Promise<DataRow[]> {
  const result = await getWorldBankData("SP.POP.TOTL", country, from, to);
  return result.rows;
}
