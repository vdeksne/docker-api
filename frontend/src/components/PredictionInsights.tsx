import React from "react";

type InsightPayload = {
  overview?: string;
  by_country?: Array<{
    country: string;
    summary: string;
    change_pct?: number;
    base_value?: number;
    predicted_value?: number;
    data_points?: number;
  }>;
  comparative?: string[];
  drivers?: string[];
  risks?: string[];
  scenarios?: string[];
  policy?: string[];
  communication?: string[];
  caveats?: string[];
  target_year?: number;
};

type PredictionInsightsProps = {
  indicatorCode: string;
  indicatorName: string;
  insights: InsightPayload;
  getCountryName: (code: string) => string;
};

const SECTION_STYLE: React.CSSProperties = {
  background: "rgba(25, 118, 210, 0.08)",
  border: "1px solid rgba(144, 202, 249, 0.12)",
  borderRadius: 12,
  padding: "18px 20px",
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

const SECTION_TITLE_STYLE: React.CSSProperties = {
  color: "#90caf9",
  fontSize: 14,
  fontWeight: 700,
  letterSpacing: "0.04em",
  textTransform: "uppercase",
  margin: 0,
};

const ITEM_TEXT_STYLE: React.CSSProperties = {
  color: "#e3f2fd",
  fontSize: 14,
  fontWeight: 500,
  lineHeight: 1.5,
  margin: 0,
};

export function PredictionInsights({
  indicatorCode,
  indicatorName,
  insights,
  getCountryName,
}: PredictionInsightsProps) {
  if (!insights) return null;

  const sections: Array<{
    key: string;
    title: string;
    items: string[];
  }> = [];

  if (insights.comparative && insights.comparative.length > 0) {
    sections.push({
      key: "comparative",
      title: "Comparative Highlights",
      items: insights.comparative,
    });
  }
  if (insights.drivers && insights.drivers.length > 0) {
    sections.push({
      key: "drivers",
      title: "Potential Drivers",
      items: insights.drivers,
    });
  }
  if (insights.risks && insights.risks.length > 0) {
    sections.push({
      key: "risks",
      title: "Risk Alerts",
      items: insights.risks,
    });
  }
  if (insights.scenarios && insights.scenarios.length > 0) {
    sections.push({
      key: "scenarios",
      title: "Scenario Watch",
      items: insights.scenarios,
    });
  }
  if (insights.policy && insights.policy.length > 0) {
    sections.push({
      key: "policy",
      title: "Policy Suggestions",
      items: insights.policy,
    });
  }
  if (insights.communication && insights.communication.length > 0) {
    sections.push({
      key: "communication",
      title: "Communication Hooks",
      items: insights.communication,
    });
  }
  if (insights.caveats && insights.caveats.length > 0) {
    sections.push({
      key: "caveats",
      title: "Data Caveats",
      items: insights.caveats,
    });
  }

  return (
    <div
      style={{
        background: "rgba(10, 25, 41, 0.85)",
        border: "1px solid rgba(144, 202, 249, 0.16)",
        borderRadius: 16,
        padding: "24px 28px",
        boxShadow: "0 10px 35px rgba(0, 0, 0, 0.35)",
        display: "flex",
        flexDirection: "column",
        gap: 20,
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <span
          style={{
            color: "rgba(144, 202, 249, 0.8)",
            fontSize: 13,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            fontWeight: 700,
          }}
        >
          AI Narrative
        </span>
        <h3
          style={{
            margin: 0,
            color: "#e3f2fd",
            fontSize: 24,
            fontWeight: 700,
            letterSpacing: "-0.3px",
          }}
        >
          {indicatorName}
        </h3>
        {insights.overview && (
          <p
            style={{
              margin: 0,
              color: "rgba(227, 242, 253, 0.85)",
              fontSize: 15,
              lineHeight: 1.6,
            }}
          >
            {insights.overview}
          </p>
        )}
      </div>

      {insights.by_country && insights.by_country.length > 0 && (
        <div style={SECTION_STYLE}>
          <h4 style={SECTION_TITLE_STYLE}>Country Trend Summaries</h4>
          {insights.by_country.map((entry) => {
            const name = getCountryName(entry.country);
            const pct =
              typeof entry.change_pct === "number"
                ? `${(entry.change_pct * 100).toFixed(2)}%`
                : "0%";
            return (
              <div key={entry.country} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span
                  style={{
                    color: "#e3f2fd",
                    fontSize: 15,
                    fontWeight: 700,
                  }}
                >
                  {name} · {pct}
                </span>
                <p style={ITEM_TEXT_STYLE}>{entry.summary}</p>
              </div>
            );
          })}
        </div>
      )}

      {sections.map((section) => (
        <div key={section.key} style={SECTION_STYLE}>
          <h4 style={SECTION_TITLE_STYLE}>{section.title}</h4>
          {section.items.map((text, idx) => (
            <p key={`${section.key}-${idx}`} style={ITEM_TEXT_STYLE}>
              {text}
            </p>
          ))}
        </div>
      ))}
    </div>
  );
}
