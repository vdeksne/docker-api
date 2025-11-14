const DEFAULT_API_BASE =
  typeof window !== "undefined" && window.location.hostname !== "localhost"
    ? "https://139-59-138-164.sslip.io"
    : "http://localhost:4000";

const API_BASE =
  (import.meta as any).env?.VITE_BACKEND_URL?.trim() || DEFAULT_API_BASE;

export type CountryInsight = {
  country: string;
  summary: string;
  change_pct?: number;
  base_value?: number;
  predicted_value?: number;
  base_year?: number;
  target_year?: number;
};

export type PredictionInsights = {
  overview: string;
  by_country: CountryInsight[];
  notes?: string[];
};

export async function fetchWorldBankData({
  country,
  indicator,
  from,
  to,
}: {
  country: string;
  indicator?: string;
  from?: number;
  to?: number;
}): Promise<{
  country: string;
  indicator: string;
  indicatorName: string;
  from?: number;
  to?: number;
  rows: Array<{ year: number; value: number | null }>;
}> {
  const params = new URLSearchParams();
  params.set("country", country);
  if (indicator) params.set("indicator", indicator);
  if (from) params.set("from", String(from));
  if (to) params.set("to", String(to));
  const res = await fetch(`${API_BASE}/api/population?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return res.json();
}

// Backwards compatibility
export async function fetchPopulation({
  country,
  from,
  to,
}: {
  country: string;
  from?: number;
  to?: number;
}) {
  return fetchWorldBankData({ country, indicator: "SP.POP.TOTL", from, to });
}

/**
 * Predict population growth for a country
 */
export async function predictPopulation({
  country,
  years_ahead = 5,
  base_year,
}: {
  country: string;
  years_ahead?: number;
  base_year?: number;
}): Promise<{
  country: string;
  base_year?: number;
  target_year?: number;
  predictions: Record<number, number>;
  source?: string;
  notes?: string;
  insights?: PredictionInsights;
}> {
  const params = new URLSearchParams();
  params.set("country", country);
  params.set("years_ahead", String(years_ahead));
  if (base_year) params.set("base_year", String(base_year));

  const res = await fetch(
    `${API_BASE}/api/predictions/population?${params.toString()}`
  );
  if (!res.ok) {
    throw new Error(`Prediction request failed: ${res.status}`);
  }
  return res.json();
}

/**
 * Predict migration flows between countries
 */
export async function predictMigration({
  countries,
  target_year,
  base_year,
}: {
  countries: string[];
  target_year?: number;
  base_year?: number;
}): Promise<{
  countries: string[];
  target_year?: number;
  predictions: Record<string, number>;
}> {
  const res = await fetch(`${API_BASE}/api/predictions/migration`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      countries,
      target_year,
      base_year,
    }),
  });

  if (!res.ok) {
    throw new Error(`Migration prediction request failed: ${res.status}`);
  }
  return res.json();
}

/**
 * Predict any World Bank indicator for multiple countries
 */
export async function predictIndicator({
  indicator,
  countries,
  target_year,
  base_year,
}: {
  indicator: string;
  countries: string[];
  target_year?: number;
  base_year?: number;
}): Promise<{
  indicator: string;
  countries: string[];
  target_year?: number;
  base_year?: number;
  predictions: Record<string, number>;
  source?: string;
  notes?: string;
  insights?: PredictionInsights;
}> {
  const res = await fetch(`${API_BASE}/api/predictions/indicator`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      indicator,
      countries,
      target_year,
      base_year,
    }),
  });

  if (!res.ok) {
    throw new Error(`Indicator prediction request failed: ${res.status}`);
  }
  return res.json();
}
