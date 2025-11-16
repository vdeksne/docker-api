import React, { useState, useEffect, useRef, useCallback } from "react";
import type { PredictionInsights, CountryInsight } from "../services/api";
// @ts-ignore
import styles from "./PopulationTable.module.css";

type PopulationRow = { year: number; value: number | null };
type CountryData = {
  country: string;
  indicator: string;
  indicatorName: string;
  rows: PopulationRow[];
};

type PredictionDetail = {
  source?: string;
  notes?: string;
  targetYear: number;
  insights?: PredictionInsights;
  countryInsight?: CountryInsight | null;
};

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function toFiniteNumber(value: unknown): number | null {
  const num = typeof value === "number" ? value : Number(value);
  return Number.isFinite(num) ? num : null;
}

function isNonZeroNumber(value: number | null): value is number {
  return value !== null && value !== 0;
}

function getMinMax(arr: number[]): [number, number] {
  const numeric = arr.filter((value) => Number.isFinite(value));
  if (numeric.length === 0) return [0, 1];
  let min = Math.min(...numeric);
  let max = Math.max(...numeric);
  // Avoid collapsing to a single line if all values are the same
  if (min === max) {
    min = min - 1;
    max = max + 1;
  }
  return [min, max];
}

const COUNTRY_COLORS = [
  {
    line: "#64b5f6",
    dot: "#e3f2fd",
    dotStroke: "#64b5f6",
    gradient: ["#64b5f6", "#90caf9", "#bbdefb"],
  },
  {
    line: "#ef5350",
    dot: "#e3f2fd",
    dotStroke: "#ef5350",
    gradient: ["#ef5350", "#ff8a80", "#ffb3b3"],
  },
  {
    line: "#66bb6a",
    dot: "#e3f2fd",
    dotStroke: "#66bb6a",
    gradient: ["#66bb6a", "#81c784", "#a5d6a7"],
  },
  {
    line: "#ffa726",
    dot: "#e3f2fd",
    dotStroke: "#ffa726",
    gradient: ["#ffa726", "#ffb74d", "#ffcc80"],
  },
  {
    line: "#ab47bc",
    dot: "#e3f2fd",
    dotStroke: "#ab47bc",
    gradient: ["#ab47bc", "#ba68c8", "#ce93d8"],
  },
];

const CHART_COLORS = {
  bg: "rgba(13, 71, 161, 0.2)",
  axis: "#90caf9",
  grid: "rgba(144, 202, 249, 0.2)",
  label: "#e3f2fd",
};

const POPULATION_INDICATOR_CODES = [
  "SP.POP.TOTL",
  "SP.POP.TOTL.IN.ZS",
  "SP.POP.TOTL.FE.IN",
  "SP.POP.TOTL.MA.IN",
];
const POPULATION_INDICATOR_NAMES = [
  "Population, total",
  "Total population",
  "Population total",
  "Population",
];

function isPopulationIndicator(indicator: string, indicatorName?: string) {
  return (
    POPULATION_INDICATOR_CODES.includes(indicator) ||
    (indicatorName &&
      POPULATION_INDICATOR_NAMES.some((name) =>
        indicatorName.toLowerCase().includes(name.toLowerCase())
      ))
  );
}

// Helper function for determining Y axis label
function getYAxisLabel(processedData: any[]): string {
  if (processedData.length === 0) return "";
  const indicators = Array.from(
    new Set(processedData.map((d) => d.indicatorName))
  );
  if (indicators.length === 1) {
    return indicators[0] || "Value";
  }
  return "Value";
}

const X_AXIS_LABEL = "Year";

function selectSuggestion(options: string[], seed: string): string {
  if (options.length === 0) {
    return "Monitor underlying drivers and adjust plans quarterly.";
  }
  const hash = seed
    .split("")
    .reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return options[hash % options.length];
}

export function PopulationTable({
  countriesData,
  predictions = {},
  predictionDetails = {},
}: {
  countriesData: CountryData[];
  predictions?: Record<string, Record<number, number>>;
  predictionDetails?: Record<string, PredictionDetail>;
}) {
  // All hooks must be called at the top level, before any conditional returns
  const [tooltip, setTooltip] = useState<null | {
    x: number;
    y: number;
    year: number;
    value: number;
    country: string;
    indicatorName: string;
  }>(null);

  // Modern wide chart with zoom support
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [containerWidth, setContainerWidth] = useState(1400);
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const handleWheel = useCallback(
    (event: WheelEvent) => {
      const svgElement = svgRef.current;
      if (!svgElement) return;

      event.preventDefault();
      const rect = svgElement.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;

      const delta = event.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(0.5, Math.min(3, zoomLevel * delta));

      // Zoom towards mouse position
      const zoomChange = newZoom / zoomLevel;
      const newPanX = mouseX - (mouseX - panX) * zoomChange;
      const newPanY = mouseY - (mouseY - panY) * zoomChange;

      setZoomLevel(newZoom);
      setPanX(newPanX);
      setPanY(newPanY);
    },
    [zoomLevel, panX, panY]
  );

  useEffect(() => {
    const svgElement = svgRef.current;
    if (!svgElement) return;

    const listener = (event: WheelEvent) => handleWheel(event);
    svgElement.addEventListener("wheel", listener, { passive: false });
    return () => {
      svgElement.removeEventListener("wheel", listener);
    };
  }, [handleWheel]);

  // Responsive width - use max available width
  useEffect(() => {
    const updateWidth = () => {
      const mobile = window.innerWidth <= 768;
      setIsMobile(mobile);
      setContainerWidth(
        mobile
          ? Math.min(window.innerWidth - 24, 1400)
          : Math.min(window.innerWidth - 100, 1400)
      );
    };
    updateWidth();
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  if (!countriesData || countriesData.length === 0) {
    return (
      <p
        style={{
          color: "#90caf9",
          background: "rgba(25, 118, 210, 0.08)",
          borderRadius: 12,
          padding: "1.5em 2em",
          marginTop: 24,
          fontStyle: "italic",
          border: "1px solid rgba(144, 202, 249, 0.12)",
          fontFamily: "'Lato', sans-serif",
          fontSize: "14px",
        }}
      >
        No countries selected. Add countries to compare data.
      </p>
    );
  }

  const processedData = countriesData
    .map((cd, idx) => {
      const sorted = [...cd.rows]
        .map((r) => {
          const cleanedValue = toFiniteNumber(r.value);
          const cleanedYear = toFiniteNumber(r.year);
          if (!isNonZeroNumber(cleanedValue) || cleanedYear === null) {
            return null;
          }
          return { year: cleanedYear, value: cleanedValue as number };
        })
        .filter((r): r is { year: number; value: number } => r !== null)
        .sort((a, b) => a!.year - b!.year);
      return {
        country: cd.country,
        indicator: cd.indicator,
        indicatorName: cd.indicatorName,
        sorted,
        color: COUNTRY_COLORS[idx % COUNTRY_COLORS.length],
      };
    })
    .filter((d) => d.sorted.length > 0);

  // Debug: Log processed data to help diagnose issues
  if (processedData.length > 0) {
    console.log(
      "Processed data for chart:",
      processedData.map((d) => ({
        country: d.country,
        dataPoints: d.sorted.length,
        firstValue: d.sorted[0]?.value,
        lastValue: d.sorted[d.sorted.length - 1]?.value,
      }))
    );
  }

  if (processedData.length === 0) {
    return (
      <p
        style={{
          color: "#ff8a80",
          background: "rgba(244, 67, 54, 0.15)",
          borderRadius: 12,
          padding: "1.5em 2em",
          marginTop: 24,
          fontStyle: "italic",
          border: "1px solid rgba(244, 67, 54, 0.3)",
          fontFamily: "'Lato', sans-serif",
          fontSize: "14px",
        }}
      >
        No data to graph.
      </p>
    );
  }

  // Get unique indicator names for title
  const uniqueIndicators = Array.from(
    new Set(processedData.map((d) => d.indicatorName))
  );
  const chartTitle =
    uniqueIndicators.length === 1
      ? uniqueIndicators[0]
      : `${uniqueIndicators.length} Indicators Comparison`;

  const yAxisLabel = getYAxisLabel(processedData);

  const allYears = new Set<number>();
  const allValues: number[] = [];
  processedData.forEach((d) => {
    d.sorted.forEach((r) => {
      if (isFiniteNumber(r.year)) {
        allYears.add(r.year);
      }
      if (isFiniteNumber(r.value) && r.value !== 0) {
        allValues.push(r.value);
      }
    });
  });

  // Add prediction years and values to the sets
  Object.values(predictions).forEach((predData) => {
    Object.entries(predData).forEach(([yearStr, rawValue]) => {
      const yearNum = toFiniteNumber(yearStr);
      const valueNum = toFiniteNumber(rawValue);
      if (yearNum !== null) {
        allYears.add(yearNum);
      }
      if (valueNum !== null && valueNum !== 0) {
        allValues.push(valueNum);
      }
    });
  });

  const years = Array.from(allYears)
    .filter((year) => Number.isFinite(year))
    .sort((a, b) => a - b);
  const [minYear, maxYear] = getMinMax(years);
  const [minValue, maxValue] = getMinMax(allValues);

  const width = containerWidth - (isMobile ? 24 : 48); // Account for padding
  const height = isMobile ? 350 : 500; // Smaller height on mobile
  const paddingLeft = isMobile ? 50 : 90;
  const paddingBottom = isMobile ? 50 : 60;
  const paddingTop = isMobile ? 40 : 50;
  const paddingRight = isMobile ? 30 : 60;

  const x = (year: number) =>
    paddingLeft +
    ((year - minYear) / (maxYear - minYear || 1)) *
      (width - paddingLeft - paddingRight);
  const y = (pop: number) =>
    paddingTop +
    (1 - (pop - minValue) / (maxValue - minValue || 1)) *
      (height - paddingBottom - paddingTop);

  const xTicks =
    years.length > 18
      ? years.filter(
          (_, i) =>
            i === 0 ||
            i === years.length - 1 ||
            i % Math.ceil(years.length / 12) === 0
        )
      : years;

  const yTicksCount = 7;
  const yTicks: number[] = [];
  for (let i = 0; i < yTicksCount; ++i) {
    yTicks.push(
      Math.round(minValue + ((maxValue - minValue) * i) / (yTicksCount - 1))
    );
  }
  yTicks.reverse();

  // Process predictions and merge with historical data
  const dataWithPredictions = processedData.map((d) => {
    const predictionKey = `${d.country}-${d.indicator}`;
    const countryPredictions = predictions[predictionKey] || {};

    // Get the last historical data point
    const lastPoint = d.sorted[d.sorted.length - 1];
    const predictionPoints: Array<{ year: number; value: number }> = [];

    // Add prediction points
    if (lastPoint && Object.keys(countryPredictions).length > 0) {
      const predictionYears = Object.entries(countryPredictions)
        .map(([year, value]) => ({
          year: toFiniteNumber(year),
          value: toFiniteNumber(value),
        }))
        .filter(
          (
            p
          ): p is {
            year: number;
            value: number;
          } => p.year !== null && p.value !== null
        )
        .sort((a, b) => a.year - b.year);

      predictionYears.forEach((predPoint) => {
        if (predPoint.year > lastPoint.year && predPoint.value !== 0) {
          predictionPoints.push({
            year: predPoint.year,
            value: predPoint.value,
          });
        }
      });
    }

    return {
      ...d,
      predictionPoints,
    };
  });

  const tableSeries = dataWithPredictions.map((d) => {
    const key = `${d.country}-${d.indicator}`;
    const valueMap = new Map<
      number,
      { value: number; isPrediction: boolean }
    >();

    d.sorted.forEach((r) => {
      valueMap.set(r.year, {
        value: r.value as number,
        isPrediction: false,
      });
    });

    d.predictionPoints.forEach((p) => {
      if (p.value !== 0) {
        valueMap.set(p.year, {
          value: p.value,
          isPrediction: true,
        });
      }
    });

    return {
      key,
      label: `${d.country} - ${d.indicatorName}`,
      color: d.color,
      values: valueMap,
    };
  });

  const tableYears = years;

  const insightOverviewTexts: string[] = [];

  const insightCards = tableSeries
    .map((series) => {
      const detail = predictionDetails[series.key];
      if (!detail) return null;
      const insightSummary =
        detail.countryInsight?.summary ||
        detail.insights?.overview ||
        "Forecast derived from historical trend analysis.";
      if (detail.insights?.overview) {
        insightOverviewTexts.push(detail.insights.overview);
      }
      const changePct = detail.countryInsight?.change_pct ?? null;
      let actionSuggestion =
        "Monitor underlying drivers and adjust plans quarterly.";
      if (typeof changePct === "number") {
        if (changePct > 0.01) {
          actionSuggestion = selectSuggestion(
            [
              "Launch a growth sprint: scale housing, mobility, and workforce pipelines to absorb the surge.",
              "Pair forecasted expansion with bold infrastructure upgrades and talent acquisition blitzes.",
              "Channel growth into innovation districts; capture momentum with fast-track development zones.",
            ],
            `${series.key}-rapid`
          );
        } else if (changePct > 0.002) {
          actionSuggestion = selectSuggestion(
            [
              "Stay ahead of the curve—tune education, healthcare, and transit capacity before bottlenecks form.",
              "Activate mid-scale growth investments and monitor migration dashboards monthly.",
              "Seed co-working hubs and affordable housing pilots to gently accelerate positive momentum.",
            ],
            `${series.key}-steady-up`
          );
        } else if (changePct < -0.01) {
          actionSuggestion = selectSuggestion(
            [
              "Mobilize a resilience taskforce: attract return migrants, automate services, and boost productivity.",
              "Offer relocation incentives plus business tax relief to counter sharp demographic outflows.",
              "Stabilize the base with childcare credits, re-skilling stipends, and targeted civic engagement campaigns.",
            ],
            `${series.key}-steep-down`
          );
        } else if (changePct < -0.002) {
          actionSuggestion = selectSuggestion(
            [
              "Deploy micro-incentives—startup grants, cultural assets, and quality-of-life perks—to flatten decline.",
              "Tune immigration pathways and strengthen university-to-work pipelines to cushion soft contraction.",
              "Map vulnerable communities and deliver targeted support before the trend deepens.",
            ],
            `${series.key}-soft-down`
          );
        } else {
          actionSuggestion = selectSuggestion(
            [
              "Use this plateau to modernize services, stress-test scenarios, and lock in resilience wins.",
              "Maintain a watchful equilibrium—calibrate small policy nudges and keep analytics on standby.",
              "Convert stability into opportunity: pilot data-driven civic programs while pressure is low.",
            ],
            `${series.key}-flat`
          );
        }
      }

      return {
        key: series.key,
        label: series.label,
        color: series.color,
        source: detail.source || "ml-service",
        notes: detail.notes,
        insightSummary,
        changePct,
        targetYear: detail.targetYear,
        extraNotes: detail.insights?.notes ?? [],
        trendLabel:
          changePct && changePct > 0.005
            ? "Rising"
            : changePct && changePct < -0.005
            ? "Falling"
            : "Stable",
        action: actionSuggestion,
      };
    })
    .filter((card): card is NonNullable<typeof card> => card !== null);

  const uniqueOverviewTexts = Array.from(new Set(insightOverviewTexts));

  const yoySeries = tableSeries.map((series) => {
    const sortedYears = Array.from(series.values.keys()).sort((a, b) => a - b);
    const data: Array<{ year: number; change: number }> = [];
    for (let i = 1; i < sortedYears.length; i++) {
      const year = sortedYears[i];
      const prevYear = sortedYears[i - 1];
      const current = series.values.get(year);
      const previous = series.values.get(prevYear);
      if (!current || !previous || previous.value === 0) continue;
      const change = (current.value - previous.value) / previous.value;
      data.push({ year, change });
    }
    return {
      label: series.label,
      color: series.color,
      data,
    };
  });

  const yoyAllChanges = yoySeries.flatMap((series) =>
    series.data.map((d) => d.change * 100)
  );
  const hasYoyData = yoyAllChanges.length > 0;
  const [yoyMin, yoyMax] = hasYoyData
    ? getMinMax(yoyAllChanges.map((val) => Math.round(val * 1000) / 1000))
    : [-2, 2];
  const yoyYears = Array.from(
    new Set(yoySeries.flatMap((s) => s.data.map((d) => d.year)))
  ).sort((a, b) => a - b);
  const yoyWidth = containerWidth - 68;
  const yoyHeight = 220;
  const yoyPaddingLeft = 80;
  const yoyPaddingRight = 50;
  const yoyPaddingTop = 40;
  const yoyPaddingBottom = isMobile ? 60 : 70; // Increased for vertical labels
  const yoyX = (year: number) =>
    yoyPaddingLeft +
    ((year - (yoyYears[0] ?? year)) /
      ((yoyYears[yoyYears.length - 1] ?? year) - (yoyYears[0] ?? year) || 1)) *
      (yoyWidth - yoyPaddingLeft - yoyPaddingRight);
  const yoyY = (value: number) =>
    yoyPaddingTop +
    (1 - (value - yoyMin) / (yoyMax - yoyMin || 1)) *
      (yoyHeight - yoyPaddingBottom - yoyPaddingTop);
  const yoyPaths = yoySeries
    .map((series) => {
      if (series.data.length === 0) return null;
      const segments = series.data
        .map((point, idx) => {
          const X = yoyX(point.year);
          const Y = yoyY(point.change * 100);
          if (!Number.isFinite(X) || !Number.isFinite(Y)) return null;
          return `${idx === 0 ? "M" : "L"}${X},${Y}`;
        })
        .filter((segment): segment is string => segment !== null);
      if (segments.length === 0) return null;
      return {
        label: series.label,
        color: series.color,
        path: segments.join(" "),
      };
    })
    .filter((entry): entry is NonNullable<typeof entry> => entry !== null);
  const yoyTickCount = 6;
  const yoyTicks = hasYoyData
    ? Array.from(
        { length: yoyTickCount },
        (_, idx) =>
          yoyMax - ((yoyMax - yoyMin) * idx) / Math.max(1, yoyTickCount - 1)
      )
    : [];

  // Use the same scales for both historical and predictions (already calculated above)
  const xWithPred = x;
  const yWithPred = y;

  const countryPaths = processedData
    .map((d) => {
      const segments = d.sorted
        .map((r, i) => {
          const X = x(r.year);
          const Y = y(r.value as number);
          if (!Number.isFinite(X) || !Number.isFinite(Y)) {
            return null;
          }
          return `${i === 0 ? "M" : "L"}${X},${Y}`;
        })
        .filter((segment): segment is string => segment !== null);

      if (segments.length === 0) {
        return null;
      }

      return {
        country: d.country,
        indicator: d.indicator,
        path: segments.join(" "),
        color: d.color,
      };
    })
    .filter((entry): entry is NonNullable<typeof entry> => entry !== null);

  // Create prediction paths (dashed lines)
  const predictionPaths = dataWithPredictions
    .filter((d) => d.predictionPoints.length > 0)
    .map((d, idx) => {
      const lastPoint = d.sorted[d.sorted.length - 1];
      if (!lastPoint) return null;

      const startX = x(lastPoint.year);
      const startY = y(lastPoint.value as number);
      if (!Number.isFinite(startX) || !Number.isFinite(startY)) {
        return null;
      }

      const segments = d.predictionPoints
        .map((p, i) => {
          const X = xWithPred(p.year);
          const Y = yWithPred(p.value);
          if (!Number.isFinite(X) || !Number.isFinite(Y)) {
            return null;
          }
          return i === 0 ? `M ${startX},${startY} L ${X},${Y}` : `L ${X},${Y}`;
        })
        .filter((segment): segment is string => segment !== null);

      if (segments.length === 0) {
        return null;
      }

      return {
        country: d.country,
        indicator: d.indicator,
        path: segments.join(" "),
        color: d.color,
        predictionPoints: d.predictionPoints,
        lastPoint,
      };
    })
    .filter((p): p is NonNullable<typeof p> => p !== null);

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.button === 0 && !e.ctrlKey && !e.metaKey) {
      // Left mouse button, not with modifier
      setIsDragging(true);
      setDragStart({ x: e.clientX - panX, y: e.clientY - panY });
      e.preventDefault();
    }
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (isDragging) {
      setPanX(e.clientX - dragStart.x);
      setPanY(e.clientY - dragStart.y);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const resetZoom = () => {
    setZoomLevel(1);
    setPanX(0);
    setPanY(0);
  };

  return (
    <div
      style={{
        marginTop: isMobile ? 20 : 32,
        background: "rgba(25, 118, 210, 0.08)",
        border: "1px solid rgba(144, 202, 249, 0.12)",
        borderRadius: 16,
        padding: isMobile ? "20px 12px 16px 12px" : "32px 24px 24px 24px",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
        width: "100%",
        maxWidth: "100%",
        marginLeft: "auto",
        marginRight: "auto",
        fontFamily: "'Lato', sans-serif",
        overflowX: "auto" /* Allow horizontal scroll on mobile */,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 24,
        }}
      >
        <h3
          style={{
            margin: 0,
            fontWeight: 700,
            letterSpacing: "-0.3px",
            fontSize: isMobile ? 20 : 28,
            color: "#e3f2fd",
            fontFamily: "'Lato', sans-serif",
          }}
        >
          {chartTitle}
        </h3>
        <div
          style={{
            display: "flex",
            gap: 12,
            alignItems: "center",
            flexWrap: isMobile ? "wrap" : "nowrap",
          }}
        >
          <button
            onClick={() => setZoomLevel(Math.max(0.5, zoomLevel - 0.1))}
            style={{
              padding: "8px 16px",
              background: "rgba(144, 202, 249, 0.15)",
              color: "#90caf9",
              border: "1px solid rgba(144, 202, 249, 0.3)",
              borderRadius: 8,
              cursor: "pointer",
              fontSize: "14px",
              fontFamily: "'Lato', sans-serif",
              fontWeight: 600,
            }}
          >
            −
          </button>
          <span
            style={{
              color: "#90caf9",
              fontSize: "14px",
              fontFamily: "'Lato', sans-serif",
              minWidth: "60px",
              textAlign: "center",
            }}
          >
            {Math.round(zoomLevel * 100)}%
          </span>
          <button
            onClick={() => setZoomLevel(Math.min(3, zoomLevel + 0.1))}
            style={{
              padding: "8px 16px",
              background: "rgba(144, 202, 249, 0.15)",
              color: "#90caf9",
              border: "1px solid rgba(144, 202, 249, 0.3)",
              borderRadius: 8,
              cursor: "pointer",
              fontSize: "14px",
              fontFamily: "'Lato', sans-serif",
              fontWeight: 600,
            }}
          >
            +
          </button>
          <button
            onClick={resetZoom}
            style={{
              padding: "8px 16px",
              background: "rgba(144, 202, 249, 0.15)",
              color: "#90caf9",
              border: "1px solid rgba(144, 202, 249, 0.3)",
              borderRadius: 8,
              cursor: "pointer",
              fontSize: "13px",
              fontFamily: "'Lato', sans-serif",
              fontWeight: 600,
              marginLeft: 8,
            }}
          >
            Reset
          </button>
        </div>
      </div>
      <div
        style={{
          overflow: "hidden",
          width: "100%",
          paddingBottom: 2,
          borderRadius: 12,
          // background: "rgba(13, 71, 161, 0.25)",
          border: "1px solid rgba(144, 202, 249, 0.25)",
          position: "relative",
        }}
      >
        <svg
          ref={svgRef}
          width={width}
          height={height}
          style={{
            background: CHART_COLORS.bg,
            display: "block",
            cursor: isDragging ? "grabbing" : "grab",
            shapeRendering: "geometricPrecision",
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <g transform={`translate(${panX}, ${panY}) scale(${zoomLevel})`}>
            <defs>
              {COUNTRY_COLORS.map((color, idx) => (
                <linearGradient
                  key={`gradient-${idx}`}
                  id={`line-gradient-${idx}`}
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop offset="0%" stopColor={color.gradient[0]} />
                  <stop offset="40%" stopColor={color.gradient[1]} />
                  <stop offset="100%" stopColor={color.gradient[2]} />
                </linearGradient>
              ))}
              <filter id="shadow" x="-5" y="-5" width="180" height="180">
                <feDropShadow
                  dx="0.5"
                  dy="2"
                  stdDeviation="1.5"
                  floodColor="rgba(144, 202, 249, 0.3)"
                />
              </filter>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            {/* Y axis */}
            <line
              x1={paddingLeft}
              y1={paddingTop}
              x2={paddingLeft}
              y2={height - paddingBottom}
              stroke={CHART_COLORS.axis}
              strokeWidth={2.5}
              opacity={1}
            />
            {/* X axis */}
            <line
              x1={paddingLeft}
              y1={height - paddingBottom}
              x2={width - paddingRight}
              y2={height - paddingBottom}
              stroke={CHART_COLORS.axis}
              strokeWidth={2.5}
              opacity={1}
            />
            {/* Y axis ticks & labels */}
            {yTicks.map((tick, i) => {
              const tickY = y(tick);
              if (!Number.isFinite(tickY)) {
                return null;
              }
              return (
                <g key={tick}>
                  <line
                    x1={paddingLeft - 11}
                    y1={tickY}
                    x2={width - paddingRight}
                    y2={tickY}
                    stroke={CHART_COLORS.grid}
                    strokeDasharray="3 7"
                  />
                  <line
                    x1={paddingLeft - 11}
                    y1={tickY}
                    x2={paddingLeft}
                    y2={tickY}
                    stroke={CHART_COLORS.axis}
                    strokeWidth={1.5}
                  />
                  <text
                    x={paddingLeft - 15}
                    y={tickY + 5}
                    fontSize={13}
                    textAnchor="end"
                    fill={CHART_COLORS.label}
                    style={{
                      fontFamily: "'Lato', sans-serif",
                      fontWeight: 600,
                      opacity: i === 0 || i === yTicks.length - 1 ? 0.7 : 0.9,
                    }}
                  >
                    {tick.toLocaleString()}
                  </text>
                </g>
              );
            })}
            {/* Y axis label - move it up, above the top yTick tick, so it's not overlapping numbers */}
            <text
              x={paddingLeft - 45}
              y={paddingTop - 12}
              textAnchor="middle"
              fontSize={15}
              fill={CHART_COLORS.label}
              style={{
                fontFamily: "'Lato', sans-serif",
                fontWeight: 700,
                opacity: 0.9,
              }}
            >
              {yAxisLabel}
            </text>
            {/* X axis ticks & labels */}
            {xTicks.map((year, i) => {
              const tickX = x(year);
              if (!Number.isFinite(tickX)) {
                return null;
              }
              return (
                <g key={year}>
                  <line
                    x1={tickX}
                    y1={height - paddingBottom}
                    x2={tickX}
                    y2={height - paddingBottom + 13}
                    stroke={CHART_COLORS.axis}
                    strokeWidth={1.5}
                  />
                  <text
                    x={tickX}
                    y={height - paddingBottom + 28}
                    fontSize={13}
                    textAnchor="middle"
                    fill={CHART_COLORS.label}
                    style={{
                      fontFamily: "'Lato', sans-serif",
                      fontWeight: 600,
                      opacity: i === 0 || i === xTicks.length - 1 ? 0.8 : 1,
                    }}
                  >
                    {year}
                  </text>
                </g>
              );
            })}
            {/* X axis label */}
            <text
              x={paddingLeft + (width - paddingLeft - paddingRight) / 2}
              y={height - 10}
              textAnchor="middle"
              fontSize={15}
              fill={CHART_COLORS.label}
              style={{
                fontFamily: "'Lato', sans-serif",
                fontWeight: 700,
                opacity: 0.9,
              }}
            >
              {X_AXIS_LABEL}
            </text>
            {/* Historical data lines for each country */}
            {countryPaths.map((cp, idx) => (
              <path
                key={`${cp.country}-${cp.indicator}-${idx}`}
                d={cp.path}
                stroke={cp.color.line}
                strokeWidth={4}
                fill="none"
                filter="url(#glow)"
                opacity={0.95}
              />
            ))}
            {/* Prediction lines (dashed) */}
            {predictionPaths.map((pp, idx) => (
              <path
                key={`pred-${pp.country}-${pp.indicator}-${idx}`}
                d={pp.path}
                stroke={pp.color.line}
                strokeWidth={4}
                strokeDasharray="10 5"
                fill="none"
                opacity={0.8}
                filter="url(#glow)"
              />
            ))}
            {/* Dots for historical data */}
            {processedData.map((d) =>
              d.sorted
                .map((r) => {
                  const dotX = x(r.year);
                  const dotY = y(r.value as number);
                  if (!Number.isFinite(dotX) || !Number.isFinite(dotY)) {
                    return null;
                  }
                  return (
                    <g key={`${d.country}-${d.indicator}-${r.year}`}>
                      <circle
                        cx={dotX}
                        cy={dotY}
                        r={8}
                        fill={d.color.dot}
                        stroke={d.color.dotStroke}
                        strokeWidth={3}
                        shapeRendering="geometricPrecision"
                        style={{
                          cursor: "pointer",
                          transition: "r 0.15s cubic-bezier(.8,2,.5,.9)",
                        }}
                        onMouseEnter={(e) => {
                          const rect = e.currentTarget.getBoundingClientRect();
                          const svgRect = (
                            e.currentTarget.ownerSVGElement as SVGSVGElement
                          )?.getBoundingClientRect();
                          setTooltip({
                            x:
                              (rect.left -
                                (svgRect?.left || 0) +
                                rect.width / 2) /
                                zoomLevel -
                              panX / zoomLevel,
                            y:
                              (rect.top -
                                (svgRect?.top || 0) +
                                rect.height / 2) /
                                zoomLevel -
                              panY / zoomLevel,
                            year: r.year,
                            value: r.value as number,
                            country: d.country,
                            indicatorName: d.indicatorName,
                          });
                        }}
                        onMouseLeave={() => setTooltip(null)}
                        tabIndex={0}
                        aria-label={`${d.country} - Year ${r.year}, ${
                          d.indicatorName
                        } ${r.value?.toLocaleString()}`}
                      />
                    </g>
                  );
                })
                .filter((node): node is JSX.Element => node !== null)
            )}
            {/* Prediction dots */}
            {predictionPaths.map((pp) =>
              pp.predictionPoints
                .map((p) => {
                  const dotX = xWithPred(p.year);
                  const dotY = yWithPred(p.value);
                  if (!Number.isFinite(dotX) || !Number.isFinite(dotY)) {
                    return null;
                  }
                  return (
                    <g key={`pred-${pp.country}-${pp.indicator}-${p.year}`}>
                      <circle
                        cx={dotX}
                        cy={dotY}
                        r={7}
                        fill={pp.color.line}
                        stroke="#0a1929"
                        strokeWidth={2.5}
                        opacity={0.95}
                        shapeRendering="geometricPrecision"
                        style={{
                          cursor: "pointer",
                        }}
                        onMouseEnter={(e) => {
                          const rect = e.currentTarget.getBoundingClientRect();
                          const svgRect = (
                            e.currentTarget.ownerSVGElement as SVGSVGElement
                          )?.getBoundingClientRect();
                          setTooltip({
                            x:
                              (rect.left -
                                (svgRect?.left || 0) +
                                rect.width / 2) /
                                zoomLevel -
                              panX / zoomLevel,
                            y:
                              (rect.top -
                                (svgRect?.top || 0) +
                                rect.height / 2) /
                                zoomLevel -
                              panY / zoomLevel,
                            year: p.year,
                            value: p.value,
                            country: pp.country,
                            indicatorName:
                              countriesData.find(
                                (cd) =>
                                  cd.country === pp.country &&
                                  cd.indicator === pp.indicator
                              )?.indicatorName || "Prediction",
                          });
                        }}
                        onMouseLeave={() => setTooltip(null)}
                        tabIndex={0}
                        aria-label={`${pp.country} - Year ${
                          p.year
                        }, Prediction ${p.value.toLocaleString()}`}
                      />
                    </g>
                  );
                })
                .filter((node): node is JSX.Element => node !== null)
            )}
            {/* Tooltip */}
            {tooltip && (
              <foreignObject
                x={tooltip.x - 100}
                y={tooltip.y - 70}
                width={200}
                height={90}
                pointerEvents="none"
              >
                <div
                  style={{
                    position: "absolute",
                    pointerEvents: "none",
                    width: 200,
                    minHeight: 85,
                    left: 0,
                    top: 0,
                    background: "rgba(13, 71, 161, 0.95)",
                    border: "1.5px solid rgba(144, 202, 249, 0.5)",
                    borderRadius: 8,
                    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.4)",
                    fontSize: 12,
                    fontWeight: 600,
                    color: "#e3f2fd",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: "10px 12px",
                    zIndex: 10,
                    fontFamily: "'Lato', sans-serif",
                    backdropFilter: "blur(10px)",
                    boxSizing: "border-box",
                    overflow: "visible",
                  }}
                >
                  <span
                    style={{
                      marginBottom: 3,
                      color: "#90caf9",
                      fontWeight: 600,
                      fontSize: 9,
                      letterSpacing: "0.01em",
                      fontFamily: "'Lato', sans-serif",
                      textAlign: "center",
                      lineHeight: 1.2,
                      maxWidth: "180px",
                      wordWrap: "break-word",
                      whiteSpace: "normal",
                    }}
                  >
                    {tooltip.country} - {tooltip.indicatorName}
                  </span>
                  <span
                    style={{
                      marginBottom: 3,
                      color: "#e3f2fd",
                      fontWeight: 700,
                      fontSize: 16,
                      letterSpacing: "0.01em",
                      fontFamily: "'Lato', sans-serif",
                      textAlign: "center",
                      lineHeight: 1.3,
                    }}
                  >
                    {tooltip.value.toLocaleString()}
                  </span>
                  <span
                    style={{
                      color: "#90caf9",
                      fontSize: 10,
                      fontWeight: 500,
                      letterSpacing: "0.01em",
                      fontFamily: "'Lato', sans-serif",
                      opacity: 0.8,
                      textAlign: "center",
                      lineHeight: 1.3,
                    }}
                  >
                    Year {tooltip.year}
                  </span>
                </div>
              </foreignObject>
            )}
          </g>
        </svg>
        <div
          style={{
            position: "absolute",
            bottom: 12,
            right: 12,
            color: "#90caf9",
            fontSize: "12px",
            fontFamily: "'Lato', sans-serif",
            opacity: 0.6,
            background: "rgba(13, 71, 161, 0.3)",
            padding: "4px 8px",
            borderRadius: 6,
            border: "1px solid rgba(144, 202, 249, 0.2)",
          }}
        >
          Scroll to zoom • Drag to pan
        </div>
      </div>
      {hasYoyData && (
        <div
          style={{
            marginTop: 28,
            background: "rgba(25, 118, 210, 0.06)",
            border: "1px solid rgba(144, 202, 249, 0.15)",
            borderRadius: 12,
            padding: "24px 20px",
            boxShadow: "0 6px 26px rgba(0, 0, 0, 0.28)",
            overflow: "scroll",
          }}
        >
          <h4
            style={{
              margin: "0 0 14px 0",
              color: "#e3f2fd",
              fontSize: 18,
              fontWeight: 600,
              letterSpacing: "-0.2px",
              fontFamily: "'Lato', sans-serif",
              overflow: "scroll",
            }}
          >
            Year-over-Year Change (Forecast vs Historical)
          </h4>
          <svg width={yoyWidth} height={yoyHeight}>
            <g>
              <line
                x1={yoyPaddingLeft}
                y1={yoyPaddingTop}
                x2={yoyPaddingLeft}
                y2={yoyHeight - yoyPaddingBottom}
                stroke="rgba(144, 202, 249, 0.6)"
                strokeWidth={1.5}
              />
              <line
                x1={yoyPaddingLeft}
                y1={yoyHeight - yoyPaddingBottom}
                x2={yoyWidth - yoyPaddingRight}
                y2={yoyHeight - yoyPaddingBottom}
                stroke="rgba(144, 202, 249, 0.6)"
                strokeWidth={1.5}
              />
              {yoyTicks.map((tick, idx) => {
                const tickY = yoyY(tick);
                return (
                  <g key={`yoy-tick-${idx}`}>
                    <line
                      x1={yoyPaddingLeft}
                      y1={tickY}
                      x2={yoyWidth - yoyPaddingRight}
                      y2={tickY}
                      stroke="rgba(144, 202, 249, 0.15)"
                      strokeDasharray="3 6"
                    />
                    <text
                      x={yoyPaddingLeft - 12}
                      y={tickY + 4}
                      textAnchor="end"
                      fontSize={12}
                      fill="rgba(227, 242, 253, 0.75)"
                      style={{ fontFamily: "'Lato', sans-serif" }}
                    >
                      {tick.toFixed(1)}%
                    </text>
                  </g>
                );
              })}
              {yoyPaths.map((series) => (
                <path
                  key={`yoy-path-${series.label}`}
                  d={series.path}
                  stroke={series.color.line}
                  strokeWidth={3}
                  fill="none"
                  opacity={0.9}
                />
              ))}
              {(isMobile
                ? yoyYears.filter(
                    (_, index) =>
                      index % 5 === 0 || index === yoyYears.length - 1
                  )
                : yoyYears
              ).map((year) => {
                const tickX = yoyX(year);
                return (
                  <g key={`yoy-year-${year}`}>
                    <line
                      x1={tickX}
                      y1={yoyHeight - yoyPaddingBottom}
                      x2={tickX}
                      y2={yoyHeight - yoyPaddingBottom + 10}
                      stroke="rgba(144, 202, 249, 0.4)"
                    />
                    <text
                      x={tickX}
                      y={yoyHeight - yoyPaddingBottom + 40}
                      textAnchor="middle"
                      fontSize={12}
                      fill="rgba(227, 242, 253, 0.75)"
                      transform={`rotate(-90, ${tickX}, ${
                        yoyHeight - yoyPaddingBottom + 40
                      })`}
                      style={{ fontFamily: "'Lato', sans-serif" }}
                    >
                      {year}
                    </text>
                  </g>
                );
              })}
              {yoyMin < 0 && yoyMax > 0 && (
                <line
                  x1={yoyPaddingLeft}
                  y1={yoyY(0)}
                  x2={yoyWidth - yoyPaddingRight}
                  y2={yoyY(0)}
                  stroke="rgba(255, 138, 128, 0.5)"
                  strokeDasharray="4 4"
                />
              )}
            </g>
          </svg>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 16,
              marginTop: 12,
            }}
          >
            {tableSeries.map((series) => (
              <div
                key={`yoy-legend-${series.key}`}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  background: "rgba(13, 71, 161, 0.15)",
                  padding: "6px 12px",
                  borderRadius: 8,
                  border: "1px solid rgba(144, 202, 249, 0.2)",
                }}
              >
                <span
                  style={{
                    width: 14,
                    height: 14,
                    borderRadius: 4,
                    background: series.color.line,
                    display: "inline-block",
                  }}
                />
                <span
                  style={{
                    color: "#e3f2fd",
                    fontSize: 13,
                    fontWeight: 600,
                    fontFamily: "'Lato', sans-serif",
                  }}
                >
                  {series.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      {tableSeries.length > 0 && tableYears.length > 0 && (
        <div
          style={{
            marginTop: 28,
            background: "rgba(25, 118, 210, 0.06)",
            border: "1px solid rgba(144, 202, 249, 0.15)",
            borderRadius: 12,
            padding: "20px 18px",
            boxShadow: "0 6px 22px rgba(0, 0, 0, 0.25)",
          }}
        >
          <h4
            style={{
              margin: "0 0 14px 0",
              color: "#e3f2fd",
              fontSize: 18,
              fontWeight: 600,
              letterSpacing: "-0.2px",
              fontFamily: "'Lato', sans-serif",
            }}
          >
            Data Table (Chart Series)
          </h4>
          <div
            style={{
              overflowX: "auto",
              WebkitOverflowScrolling: "touch" /* Smooth scrolling on iOS */,
            }}
          >
            <table
              style={{
                width: "100%",
                minWidth: isMobile ? 600 : 480,
                borderCollapse: "collapse",
                fontFamily: "'Lato', sans-serif",
                color: "#e3f2fd",
              }}
            >
              <thead>
                <tr>
                  <th
                    style={{
                      textAlign: "left",
                      padding: "10px 14px",
                      background: "rgba(13, 71, 161, 0.35)",
                      fontWeight: 700,
                      fontSize: 13,
                      letterSpacing: "0.01em",
                      borderBottom: "1px solid rgba(144, 202, 249, 0.15)",
                    }}
                  >
                    Year
                  </th>
                  {tableSeries.map((series) => (
                    <th
                      key={`head-${series.key}`}
                      style={{
                        textAlign: "left",
                        padding: "10px 14px",
                        background: "rgba(13, 71, 161, 0.35)",
                        fontWeight: 700,
                        fontSize: 13,
                        letterSpacing: "0.01em",
                        borderBottom: "1px solid rgba(144, 202, 249, 0.15)",
                      }}
                    >
                      {series.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableYears.map((year, idx) => (
                  <tr
                    key={`row-${year}`}
                    style={{
                      background:
                        idx % 2 === 0
                          ? "rgba(13, 71, 161, 0.18)"
                          : "rgba(13, 71, 161, 0.1)",
                    }}
                  >
                    <td
                      style={{
                        padding: "8px 14px",
                        fontWeight: 600,
                        fontSize: 13,
                        borderBottom: "1px solid rgba(144, 202, 249, 0.1)",
                        color: "#90caf9",
                      }}
                    >
                      {year}
                    </td>
                    {tableSeries.map((series) => {
                      const entry = series.values.get(year);
                      if (!entry) {
                        return (
                          <td
                            key={`${series.key}-${year}`}
                            style={{
                              padding: "8px 14px",
                              borderBottom:
                                "1px solid rgba(144, 202, 249, 0.06)",
                              fontSize: 13,
                              color: "rgba(227, 242, 253, 0.55)",
                              fontStyle: "italic",
                              verticalAlign: "middle",
                              textAlign: "center",
                            }}
                          >
                            —
                          </td>
                        );
                      }
                      return (
                        <td
                          key={`${series.key}-${year}`}
                          style={{
                            padding: "8px 14px",
                            borderBottom: "1px solid rgba(144, 202, 249, 0.06)",
                            fontSize: 13,
                            fontWeight: entry.isPrediction ? 600 : 500,
                            color: entry.isPrediction
                              ? series.color.line
                              : "rgba(227, 242, 253, 0.95)",
                            textAlign: "right",
                            verticalAlign: "middle",
                          }}
                        >
                          <span>{entry.value.toLocaleString()}</span>
                          {entry.isPrediction && (
                            <span
                              style={{
                                display: "inline-block",
                                marginLeft: 8,
                                fontSize: 10,
                                fontWeight: 700,
                                color: "rgba(227, 242, 253, 0.75)",
                                textTransform: "uppercase",
                                letterSpacing: "0.04em",
                                background: "rgba(13, 71, 161, 0.55)",
                                borderRadius: 999,
                                padding: "2px 6px",
                                border: "1px solid rgba(144, 202, 249, 0.35)",
                              }}
                            >
                              Forecast
                            </span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p
            style={{
              marginTop: 12,
              fontSize: 12,
              color: "rgba(227, 242, 253, 0.65)",
              fontFamily: "'Lato', sans-serif",
            }}
          >
            Forecast values align with the dashed projections in the chart.
          </p>
        </div>
      )}
      {insightCards.length > 0 && (
        <div className={styles.insightPanel}>
          <div className={styles.insightHeader}>
            <div className={styles.insightTag}>AI Insight Briefing</div>
            <p className={styles.insightSummaryText}>
              {uniqueOverviewTexts.length > 0
                ? uniqueOverviewTexts.join(" ")
                : "AI synthesized historical trajectories with forecast signals to highlight the most actionable shifts."}
            </p>
          </div>
          <div className={styles.insightGrid}>
            {insightCards.map((card) => {
              const changeValue =
                typeof card.changePct === "number"
                  ? (card.changePct * 100).toFixed(2)
                  : null;
              const changeColor =
                typeof card.changePct === "number"
                  ? card.changePct > 0
                    ? "#6ee7b7"
                    : card.changePct < 0
                    ? "#ff9a9e"
                    : "rgba(227, 242, 253, 0.85)"
                  : "rgba(227, 242, 253, 0.85)";
              const barWidth =
                typeof card.changePct === "number"
                  ? Math.min(100, Math.abs(card.changePct) * 400)
                  : 0;
              const badgeClass = `${styles.insightBadge} ${
                card.source === "fallback"
                  ? styles.insightBadgeFallback
                  : styles.insightBadgePrimary
              }`;

              return (
                <div key={`insight-${card.key}`} className={styles.insightCard}>
                  <div className={styles.insightCardHeader}>
                    <div className={styles.insightCardMeta}>
                      <span className={styles.insightCardTitle}>
                        {card.label}
                      </span>
                      <span className={styles.insightTrend}>
                        {card.trendLabel} • Target {card.targetYear}
                      </span>
                    </div>
                    <span className={badgeClass}>
                      {card.source === "fallback" ? "Trend Model" : "AI Model"}
                    </span>
                  </div>

                  <p className={styles.insightDescription}>
                    {card.insightSummary}
                  </p>

                  {typeof card.changePct === "number" && (
                    <div className={styles.insightDeltaRow}>
                      <span
                        className={styles.insightDeltaValue}
                        style={{ color: changeColor }}
                      >
                        {card.changePct > 0
                          ? "▲"
                          : card.changePct < 0
                          ? "▼"
                          : "■"}{" "}
                        {card.changePct > 0 ? "+" : ""}
                        {changeValue}%
                      </span>
                      <div className={styles.insightBarTrack}>
                        <div
                          className={styles.insightBarFill}
                          style={{
                            width: `${barWidth}%`,
                            background: changeColor,
                          }}
                        />
                      </div>
                    </div>
                  )}

                  <div className={styles.insightAction}>
                    <span className={styles.insightActionTitle}>
                      Action Idea
                    </span>
                    <span className={styles.insightActionText}>
                      {card.action}
                    </span>
                  </div>

                  {card.extraNotes.length > 0 && (
                    <ul className={styles.notesList}>
                      {card.extraNotes.map((note, idx) => (
                        <li key={`note-${card.key}-${idx}`}>{note}</li>
                      ))}
                    </ul>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
      {processedData.length > 0 && (
        <div
          style={{
            marginTop: 20,
            display: "flex",
            justifyContent: "center",
            flexWrap: "wrap",
            gap: 20,
            padding: "16px",
            background: "rgba(25, 118, 210, 0.05)",
            borderRadius: 12,
            border: "1px solid rgba(144, 202, 249, 0.1)",
          }}
        >
          {processedData.map((d, idx) => {
            const color = d.color;
            const countryName =
              countriesData.find((cd) => cd.country === d.country)?.country ||
              d.country;
            const hasPredictions = predictionPaths.some(
              (pp) => pp.country === d.country && pp.indicator === d.indicator
            );
            return (
              <div
                key={`${d.country}-${d.indicator}-${idx}`}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <div
                  style={{
                    width: 22,
                    height: 6,
                    background: color.line,
                    borderRadius: 3,
                  }}
                />
                {hasPredictions && (
                  <svg width={22} height={6} style={{ opacity: 0.7 }}>
                    <line
                      x1={0}
                      y1={3}
                      x2={22}
                      y2={3}
                      stroke={color.line}
                      strokeWidth={3}
                      strokeDasharray="4 2"
                    />
                  </svg>
                )}
                <span
                  style={{
                    fontSize: 14,
                    fontWeight: 600,
                    color: "#90caf9",
                    fontFamily: "'Lato', sans-serif",
                  }}
                >
                  {countryName} - {d.indicatorName}
                </span>
              </div>
            );
          })}
        </div>
      )}
      <div
        style={{
          marginTop: 20,
          textAlign: "center",
          color: "#90caf9",
          fontSize: 13,
          fontWeight: 500,
          fontFamily: "'Lato', sans-serif",
          opacity: 0.7,
        }}
      >
        <span
          style={{
            background: "rgba(25, 118, 210, 0.1)",
            padding: "6px 16px",
            borderRadius: 8,
            letterSpacing: "0.01em",
            border: "1px solid rgba(144, 202, 249, 0.2)",
          }}
        >
          Data source: The World Bank
        </span>
      </div>
    </div>
  );
}
