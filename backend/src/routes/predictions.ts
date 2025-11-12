import { Router, Request, Response } from "express";

import {
  getIndicatorPredictionOnDemand,
  getPopulationPredictionOnDemand,
} from "../lib/mlOnDemand.js";
import { getWorldBankData } from "../services/population.js";

export const predictionsRouter = Router();

type CountryInsight = {
  country: string;
  summary: string;
  change_pct?: number;
  base_value?: number;
  predicted_value?: number;
  base_year?: number;
  target_year?: number;
};

type InsightBundle = {
  overview: string;
  by_country: CountryInsight[];
  notes?: string[];
};

type IndicatorPredictionResult = {
  indicator: string;
  countries: string[];
  target_year: number;
  base_year?: number;
  predictions: Record<string, number>;
  source: "ml-service" | "fallback";
  notes?: string;
  insights?: InsightBundle;
};

type PopulationPredictionResult = {
  country: string;
  base_year: number;
  target_year: number;
  predictions: Record<number, number>;
  source: "ml-service" | "fallback";
  notes?: string;
  insights?: InsightBundle;
};

async function buildFallbackIndicatorPrediction(
  indicator: string,
  countries: string[],
  targetYear?: number,
  baseYear?: number
): Promise<IndicatorPredictionResult> {
  const normalizedIndicator = indicator.toUpperCase();
  const normalizedCountries = countries.map((c) => c.toUpperCase());
  const fallbackTargetYear =
    typeof targetYear === "number" ? targetYear : (baseYear ?? 2020) + 5;

  const countryResults = await Promise.all(
    normalizedCountries.map(async (country) => {
      const lookbackStart =
        typeof baseYear === "number" ? baseYear - 20 : fallbackTargetYear - 25;
      const lookbackEnd = baseYear ?? fallbackTargetYear - 1;

      const { rows } = await getWorldBankData(
        normalizedIndicator,
        country,
        lookbackStart > 0 ? lookbackStart : undefined,
        lookbackEnd > 0 ? lookbackEnd : undefined
      );

      const numericRows = rows
        .filter((row) => row.value !== null && row.value !== undefined)
        .sort((a, b) => a.year - b.year);

      if (numericRows.length === 0) {
        return {
          country,
          baseYear: baseYear,
          baseValue: 0,
          predictedValue: 0,
        };
      }

      const latest = numericRows[numericRows.length - 1];
      const actualBaseYear = latest.year;
      const baseValue = Number(latest.value ?? 0);
      const yearsAhead = fallbackTargetYear - actualBaseYear;

      if (yearsAhead <= 0) {
        return {
          country,
          baseYear: actualBaseYear,
          baseValue,
          predictedValue: baseValue,
        };
      }

      const window = numericRows.slice(-Math.min(numericRows.length, 8));
      const first = window[0];
      const last = window[window.length - 1];
      const spanYears = Math.max(1, last.year - first.year);
      let growthRate = 0;
      if (first.value && first.value !== 0) {
        const changePct =
          (Number(last.value ?? 0) - Number(first.value ?? 0)) /
          Number(first.value ?? 0);
        growthRate = changePct / spanYears;
      }

      if (indicator === "SP.POP.TOTL") {
        growthRate = Math.max(-0.015, Math.min(0.01, growthRate));
      }

      const predictedValue = Math.max(
        0,
        baseValue * Math.pow(1 + growthRate, yearsAhead)
      );

      return {
        country,
        baseYear: actualBaseYear,
        baseValue,
        predictedValue,
        growthRate,
      };
    })
  );

  const predictions: Record<string, number> = {};
  countryResults.forEach((result) => {
    predictions[result.country] = Number.isFinite(result.predictedValue)
      ? result.predictedValue
      : result.baseValue;
  });

  console.warn(
    `[Predictions] Using fallback indicator prediction for ${normalizedIndicator} (${normalizedCountries.join(
      ", "
    )})`
  );

  const byCountryInsights: CountryInsight[] = countryResults.map(
    (result) => {
      const change =
        result.baseValue === 0
          ? 0
          : (result.predictedValue - result.baseValue) / result.baseValue;
      const direction =
        change > 0.005 ? "increase" : change < -0.005 ? "decline" : "remain flat";
      const summary = `Projected to ${direction} by ${(change * 100).toFixed(
        2
      )}% between ${result.baseYear} and ${fallbackTargetYear}.`;
      return {
        country: result.country,
        summary,
        change_pct: change,
        base_value: result.baseValue,
        predicted_value: result.predictedValue,
        base_year: result.baseYear,
        target_year: fallbackTargetYear,
      };
    }
  );

  const overview = byCountryInsights
    .map((insight) => {
      if (!insight.change_pct) return `${insight.country}: stable`;
      if (insight.change_pct > 0.005) {
        return `${insight.country}: modest growth (+${(insight.change_pct * 100).toFixed(
          2
        )}%)`;
      }
      if (insight.change_pct < -0.005) {
        return `${insight.country}: decline (${(insight.change_pct * 100).toFixed(
          2
        )}%)`;
      }
      return `${insight.country}: steady`;
    })
    .join("; ");

  return {
    indicator: normalizedIndicator,
    countries: normalizedCountries,
    target_year: fallbackTargetYear,
    base_year: baseYear,
    predictions,
    source: "fallback",
    notes:
      "Fallback linear trend prediction (ML service unavailable). Values extrapolated from recent World Bank data.",
    insights: {
      overview: overview || "Predictions derived from historical trend extrapolation.",
      by_country: byCountryInsights,
      notes: [
        "Calculated using recent World Bank yearly values.",
        "Linear trend with gentle smoothing; unexpected shocks not captured.",
      ],
    },
  };
}

async function buildFallbackPopulationPrediction(
  country: string,
  yearsAhead?: number,
  baseYear?: number
): Promise<PopulationPredictionResult> {
  const normalizedCountry = country.toUpperCase();
  const effectiveBaseYear =
    typeof baseYear === "number" ? baseYear : new Date().getFullYear() - 1;
  const effectiveYearsAhead =
    typeof yearsAhead === "number" && yearsAhead > 0 ? yearsAhead : 5;
  const targetYear = effectiveBaseYear + effectiveYearsAhead;

  const lookbackStart = effectiveBaseYear - 20;
  const { rows } = await getWorldBankData(
    "SP.POP.TOTL",
    normalizedCountry,
    lookbackStart > 0 ? lookbackStart : undefined,
    effectiveBaseYear
  );

  const numericRows = rows
    .filter((row) => row.value !== null && row.value !== undefined)
    .sort((a, b) => a.year - b.year);

  if (numericRows.length === 0) {
    console.warn(
      `[Predictions] Population fallback has no data for ${normalizedCountry}`
    );
    return {
      country: normalizedCountry,
      base_year: effectiveBaseYear,
      target_year: targetYear,
      predictions: { [targetYear]: 0 },
      source: "fallback",
      notes:
        "No historical population data available; forecast defaults to zero.",
    };
  }

  const latest = numericRows[numericRows.length - 1];
  const actualBaseYear = latest.year;
  const baseValue = Number(latest.value ?? 0);

  const window = numericRows.slice(-Math.min(numericRows.length, 8));
  const first = window[0];
  const last = window[window.length - 1];
  const spanYears = Math.max(1, last.year - first.year);
  let growthRate = 0;
  if (first.value && first.value !== 0) {
    const changePct =
      (Number(last.value ?? 0) - Number(first.value ?? 0)) /
      Number(first.value ?? 0);
    growthRate = changePct / spanYears;
  }

  growthRate = Math.max(-0.012, Math.min(0.008, growthRate));

  const yearsDiff = targetYear - actualBaseYear;
  const forecast =
    yearsDiff > 0
      ? Math.max(0, baseValue * Math.pow(1 + growthRate, yearsDiff))
      : baseValue;

  console.warn(
    `[Predictions] Using fallback population prediction for ${normalizedCountry} (base ${actualBaseYear}=${baseValue}, growthRate=${(
      growthRate * 100
    ).toFixed(2)}%/yr)`
  );

  const changePct =
    baseValue === 0 ? 0 : (forecast - baseValue) / baseValue;
  const direction =
    changePct > 0.005 ? "increase" : changePct < -0.005 ? "decline" : "remain flat";

  return {
    country: normalizedCountry,
    base_year: actualBaseYear,
    target_year: targetYear,
    predictions: { [targetYear]: forecast },
    source: "fallback",
    notes:
      "Fallback linear trend based on recent population data (ML service unavailable).",
    insights: {
      overview: `Population expected to ${direction} by ${(changePct * 100).toFixed(
        2
      )}% between ${actualBaseYear} and ${targetYear}.`,
      by_country: [
        {
          country: normalizedCountry,
          summary: `Historical slope implies a ${direction}, projecting ${Math.round(
            forecast
          ).toLocaleString()} residents by ${targetYear}.`,
          change_pct: changePct,
          base_value: baseValue,
          predicted_value: forecast,
          base_year: actualBaseYear,
          target_year: targetYear,
        },
      ],
      notes: [
        "Trend derived from the last twenty years of historical World Bank data.",
        "Linear extrapolation with smoothing to avoid extreme swings.",
      ],
    },
  };
}

/**
 * GET /api/predictions/population?country=USA&years_ahead=5&base_year=2020
 * Predict population growth for a country
 */
predictionsRouter.get("/population", async (req: Request, res: Response) => {
  try {
    const country = String(req.query.country || "").toUpperCase();
    if (!country) {
      res.status(400).json({ error: "country query param required" });
      return;
    }

    const years_ahead = req.query.years_ahead
      ? Number(req.query.years_ahead)
      : 5;
    const base_year = req.query.base_year
      ? Number(req.query.base_year)
      : undefined;

    console.log(
      `[Predictions] Requesting population prediction for ${country}, years_ahead=${years_ahead}, base_year=${base_year}`
    );

    let mlResult: any | null = null;
    try {
      mlResult = await getPopulationPredictionOnDemand({
        country,
        yearsAhead: years_ahead,
        baseYear: base_year,
      });
    } catch (mlError: any) {
      console.error(
        "[Predictions] On-demand population prediction error:",
        mlError?.message || mlError
      );
    }

    if (mlResult && typeof mlResult === "object") {
      const normalizedPredictions: Record<number, number> = {};
      const entries = Object.entries(mlResult.predictions || {});
      entries.forEach(([yearKey, value]) => {
        const yearNum = Number(yearKey);
        const numericValue = Number(value);
        if (Number.isFinite(yearNum) && Number.isFinite(numericValue)) {
          normalizedPredictions[yearNum] = numericValue;
        }
      });

      if (Object.keys(normalizedPredictions).length > 0) {
        const baseYearResolved = Number.isFinite(Number(mlResult.base_year))
          ? Number(mlResult.base_year)
          : base_year ?? new Date().getFullYear() - 1;
        const targetYearResolved = Number.isFinite(Number(mlResult.target_year))
          ? Number(mlResult.target_year)
          : Math.max(
              ...Object.keys(normalizedPredictions).map((key) => Number(key))
            );

        const payload: PopulationPredictionResult = {
          country,
          base_year: baseYearResolved,
          target_year: targetYearResolved,
          predictions: normalizedPredictions,
          source: "ml-service",
          notes: mlResult.notes,
          insights: mlResult.insights,
        };

        res.json(payload);
        return;
      }
    }

    const fallback = await buildFallbackPopulationPrediction(
      country,
      years_ahead,
      base_year
    );
    res.json(fallback);
  } catch (e: any) {
    console.error("[Predictions] Error:", e.message);
    try {
      const fallback = await buildFallbackPopulationPrediction(
        String(req.query.country || ""),
        req.query.years_ahead ? Number(req.query.years_ahead) : undefined,
        req.query.base_year ? Number(req.query.base_year) : undefined
      );
      res.json(fallback);
    } catch (fallbackError: any) {
      console.error(
        "[Predictions] Population fallback failure:",
        fallbackError?.message || fallbackError
      );
      res.status(500).json({
        error: "Failed to get prediction",
        details: e?.message || "Unknown error",
      });
    }
  }
});

/**
 * POST /api/predictions/migration
 * Predict migration flows between countries
 * Body: { countries: ["USA", "MEX", "CAN"], target_year: 2025, base_year: 2020 }
 */
predictionsRouter.post("/migration", async (_req: Request, res: Response) => {
  res.status(503).json({
    error: "Migration predictions are not available in on-demand mode",
  });
});

/**
 * GET /api/predictions/indicator?indicator=NY.GDP.MKTP.CD&countries=USA,MEX,CAN&target_year=2025&base_year=2020
 * Predict any World Bank indicator for multiple countries
 */
predictionsRouter.get("/indicator", async (req: Request, res: Response) => {
  try {
    const indicator = String(req.query.indicator || "").toUpperCase();
    const countries_str = String(req.query.countries || "");

    if (!indicator) {
      res.status(400).json({ error: "indicator query param required" });
      return;
    }

    if (!countries_str) {
      res
        .status(400)
        .json({ error: "countries query param required (comma-separated)" });
      return;
    }

    const countries = countries_str
      .split(",")
      .map((c: string) => c.trim().toUpperCase());
    const target_year = req.query.target_year
      ? Number(req.query.target_year)
      : undefined;
    const base_year = req.query.base_year
      ? Number(req.query.base_year)
      : undefined;

    console.log(
      `[Predictions] Requesting indicator prediction: ${indicator} for countries: ${countries.join(
        ", "
      )}, target_year=${target_year}, base_year=${base_year}`
    );

    let mlResult: any | null = null;
    try {
      mlResult = await getIndicatorPredictionOnDemand({
        indicator,
        countries,
        targetYear: target_year,
        baseYear: base_year,
      });
    } catch (mlError: any) {
      console.error(
        "[Predictions] On-demand indicator prediction error:",
        mlError?.message || mlError
      );
    }

    if (mlResult && typeof mlResult === "object") {
      const resolvedIndicator = (mlResult.indicator || indicator).toUpperCase();
      const resolvedCountries = (mlResult.countries || countries).map((c: string) =>
        String(c).toUpperCase()
      );
      const rawPredictions =
        mlResult && typeof mlResult.predictions === "object"
          ? mlResult.predictions
          : {};

      const normalizedPredictions: Record<string, number> = {};
      Object.entries(rawPredictions).forEach(([countryKey, value]) => {
        const numeric = Number(value);
        if (Number.isFinite(numeric)) {
          normalizedPredictions[String(countryKey).toUpperCase()] = numeric;
        }
      });

      if (Object.keys(normalizedPredictions).length > 0) {
        const resolvedTargetYear = Number.isFinite(Number(mlResult.target_year))
          ? Number(mlResult.target_year)
          : target_year ?? base_year ?? 2025;
        const resolvedBaseYear = Number.isFinite(Number(mlResult.base_year))
          ? Number(mlResult.base_year)
          : base_year;

        const payload: IndicatorPredictionResult = {
          indicator: resolvedIndicator,
          countries: resolvedCountries,
          target_year: Number.isFinite(resolvedTargetYear)
            ? resolvedTargetYear
            : (typeof base_year === "number" ? base_year : 2020) + 5,
          base_year: resolvedBaseYear,
          predictions: normalizedPredictions,
          source: "ml-service",
          notes: mlResult.notes,
          insights: mlResult.insights,
        };

        res.json(payload);
        return;
      }
    }

    const fallback = await buildFallbackIndicatorPrediction(
      indicator,
      countries,
      target_year,
      base_year
    );
    res.json(fallback);
  } catch (e: any) {
    console.error("[Predictions] Error:", e?.message || e);
    res.status(500).json({
      error: "Failed to get prediction",
      details: e?.message || "Unknown error",
    });
  }
});

/**
 * POST /api/predictions/indicator
 * Predict any World Bank indicator
 * Body: { indicator: "NY.GDP.MKTP.CD", countries: ["USA", "MEX"], target_year: 2025, base_year: 2020 }
 */
predictionsRouter.post("/indicator", async (req: Request, res: Response) => {
  const { indicator, countries, target_year, base_year } = req.body ?? {};

  try {
    if (!indicator) {
      res.status(400).json({ error: "indicator required in request body" });
      return;
    }

    if (!countries || !Array.isArray(countries) || countries.length === 0) {
      res
        .status(400)
        .json({ error: "countries array required in request body" });
      return;
    }

    const normalizedIndicator = String(indicator).toUpperCase();
    const normalizedCountries = countries.map((c: string) => String(c).toUpperCase());

    console.log(
      `[Predictions] Requesting indicator prediction: ${normalizedIndicator} for countries: ${normalizedCountries.join(
        ", "
      )}, target_year=${target_year}, base_year=${base_year}`
    );

    let mlResult: any | null = null;
    try {
      mlResult = await getIndicatorPredictionOnDemand({
        indicator: normalizedIndicator,
        countries: normalizedCountries,
        targetYear: target_year,
        baseYear: base_year,
      });
    } catch (mlError: any) {
      console.error(
        "[Predictions] On-demand indicator prediction error:",
        mlError?.message || mlError
      );
    }

    if (mlResult && typeof mlResult === "object") {
      const resolvedIndicator = (mlResult.indicator || normalizedIndicator).toUpperCase();
      const resolvedCountries = (mlResult.countries || normalizedCountries).map(
        (c: string) => String(c).toUpperCase()
      );
      const rawPredictions =
        mlResult && typeof mlResult.predictions === "object"
          ? mlResult.predictions
          : {};

      const normalizedPredictions: Record<string, number> = {};
      Object.entries(rawPredictions).forEach(([countryKey, value]) => {
        const numeric = Number(value);
        if (Number.isFinite(numeric)) {
          normalizedPredictions[String(countryKey).toUpperCase()] = numeric;
        }
      });

      if (Object.keys(normalizedPredictions).length > 0) {
        const resolvedTargetYear = Number.isFinite(Number(mlResult.target_year))
          ? Number(mlResult.target_year)
          : target_year ?? base_year ?? 2025;
        const resolvedBaseYear = Number.isFinite(Number(mlResult.base_year))
          ? Number(mlResult.base_year)
          : base_year;

        const payload: IndicatorPredictionResult = {
          indicator: resolvedIndicator,
          countries: resolvedCountries,
          target_year: Number.isFinite(resolvedTargetYear)
            ? resolvedTargetYear
            : (typeof base_year === "number" ? base_year : 2020) + 5,
          base_year: resolvedBaseYear,
          predictions: normalizedPredictions,
          source: "ml-service",
          notes: mlResult.notes,
          insights: mlResult.insights,
        };

        res.json(payload);
        return;
      }
    }

    const fallback = await buildFallbackIndicatorPrediction(
      normalizedIndicator,
      normalizedCountries,
      target_year,
      base_year
    );
    res.json(fallback);
  } catch (e: any) {
    console.error("[Predictions] Error:", e?.message || e);
    try {
      const fallback = await buildFallbackIndicatorPrediction(
        indicator,
        countries,
        target_year,
        base_year
      );
      res.json(fallback);
    } catch (fallbackError: any) {
      console.error(
        "[Predictions] Fallback failure:",
        fallbackError?.message || fallbackError
      );
      res.status(500).json({
        error: "Failed to get prediction",
        details: e?.message || "Unknown error",
      });
    }
  }
});
