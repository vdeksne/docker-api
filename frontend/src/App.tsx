import React, { useMemo, useState, useEffect } from "react";
import {
  fetchWorldBankData,
  predictIndicator,
  predictPopulation,
  type PredictionInsights,
  type CountryInsight,
} from "./services/api";
import { PopulationTable } from "./components/PopulationTable";
import layoutStyles from "./App.module.css";

type CountryData = {
  country: string;
  indicator: string;
  indicatorName: string;
  rows: Array<{ year: number; value: number | null }>;
};

const INDICATORS = [
  { code: "SP.POP.TOTL", name: "Population, total" },
  { code: "NY.GDP.MKTP.CD", name: "GDP (current US$)" },
  { code: "NY.GDP.PCAP.CD", name: "GDP per capita (current US$)" },
  { code: "NY.GDP.MKTP.KD.ZG", name: "GDP growth (annual %)" },
  { code: "SP.DYN.LE00.IN", name: "Life expectancy at birth" },
  { code: "FP.CPI.TOTL.ZG", name: "Inflation, consumer prices (annual %)" },
  { code: "NE.TRD.GNFS.ZS", name: "Trade (% of GDP)" },
  {
    code: "SE.XPD.TOTL.GD.ZS",
    name: "Government expenditure on education (% of GDP)",
  },
  { code: "SH.XPD.CHEX.GD.ZS", name: "Health expenditure (% of GDP)" },
  { code: "SP.URB.TOTL.IN.ZS", name: "Urban population (% of total)" },
  { code: "CC.CO2.EMSE.EL", name: "CO2 emissions (Mt CO2 eq)" },
  { code: "AG.LND.FRST.ZS", name: "Forest area (% of land area)" },
  {
    code: "EG.USE.ELEC.KH.PC",
    name: "Electric power consumption (kWh per capita)",
  },
  { code: "IT.NET.USER.ZS", name: "Internet users (% of population)" },
  { code: "DT.DOD.DECT.GN.ZS", name: "External debt stocks (% of GNI)" },
  {
    code: "BX.KLT.DINV.WD.GD.ZS",
    name: "Foreign direct investment (% of GDP)",
  },
  { code: "SE.PRM.ENRR", name: "Primary school enrollment" },
  { code: "NY.GNS.ICTR.ZS", name: "Gross savings (% of GDP)" },
];

const COUNTRIES_LIST = [
  { code: "AFG", name: "Afghanistan" },
  { code: "ALB", name: "Albania" },
  { code: "DZA", name: "Algeria" },
  { code: "AGO", name: "Angola" },
  { code: "ARG", name: "Argentina" },
  { code: "ARM", name: "Armenia" },
  { code: "AUS", name: "Australia" },
  { code: "AUT", name: "Austria" },
  { code: "AZE", name: "Azerbaijan" },
  { code: "BGD", name: "Bangladesh" },
  { code: "BLR", name: "Belarus" },
  { code: "BEL", name: "Belgium" },
  { code: "BEN", name: "Benin" },
  { code: "BOL", name: "Bolivia" },
  { code: "BIH", name: "Bosnia and Herzegovina" },
  { code: "BWA", name: "Botswana" },
  { code: "BRA", name: "Brazil" },
  { code: "BRN", name: "Brunei" },
  { code: "BGR", name: "Bulgaria" },
  { code: "BFA", name: "Burkina Faso" },
  { code: "BDI", name: "Burundi" },
  { code: "KHM", name: "Cambodia" },
  { code: "CMR", name: "Cameroon" },
  { code: "CAN", name: "Canada" },
  { code: "CAF", name: "Central African Republic" },
  { code: "TCD", name: "Chad" },
  { code: "CHL", name: "Chile" },
  { code: "CHN", name: "China" },
  { code: "COL", name: "Colombia" },
  { code: "COM", name: "Comoros" },
  { code: "COG", name: "Congo" },
  { code: "COD", name: "Democratic Republic of the Congo" },
  { code: "CRI", name: "Costa Rica" },
  { code: "CIV", name: "Côte d'Ivoire" },
  { code: "HRV", name: "Croatia" },
  { code: "CUB", name: "Cuba" },
  { code: "CYP", name: "Cyprus" },
  { code: "CZE", name: "Czechia" },
  { code: "DNK", name: "Denmark" },
  { code: "DJI", name: "Djibouti" },
  { code: "DOM", name: "Dominican Republic" },
  { code: "ECU", name: "Ecuador" },
  { code: "EGY", name: "Egypt" },
  { code: "SLV", name: "El Salvador" },
  { code: "GNQ", name: "Equatorial Guinea" },
  { code: "ERI", name: "Eritrea" },
  { code: "EST", name: "Estonia" },
  { code: "SWZ", name: "Eswatini" },
  { code: "ETH", name: "Ethiopia" },
  { code: "FJI", name: "Fiji" },
  { code: "FIN", name: "Finland" },
  { code: "FRA", name: "France" },
  { code: "GAB", name: "Gabon" },
  { code: "GMB", name: "Gambia" },
  { code: "GEO", name: "Georgia" },
  { code: "DEU", name: "Germany" },
  { code: "GHA", name: "Ghana" },
  { code: "GRC", name: "Greece" },
  { code: "GRD", name: "Grenada" },
  { code: "GTM", name: "Guatemala" },
  { code: "GIN", name: "Guinea" },
  { code: "GNB", name: "Guinea-Bissau" },
  { code: "GUY", name: "Guyana" },
  { code: "HTI", name: "Haiti" },
  { code: "HND", name: "Honduras" },
  { code: "HUN", name: "Hungary" },
  { code: "ISL", name: "Iceland" },
  { code: "IND", name: "India" },
  { code: "IDN", name: "Indonesia" },
  { code: "IRN", name: "Iran" },
  { code: "IRQ", name: "Iraq" },
  { code: "IRL", name: "Ireland" },
  { code: "ISR", name: "Israel" },
  { code: "ITA", name: "Italy" },
  { code: "JAM", name: "Jamaica" },
  { code: "JPN", name: "Japan" },
  { code: "JOR", name: "Jordan" },
  { code: "KAZ", name: "Kazakhstan" },
  { code: "KEN", name: "Kenya" },
  { code: "KIR", name: "Kiribati" },
  { code: "KOR", name: "South Korea" },
  { code: "KWT", name: "Kuwait" },
  { code: "KGZ", name: "Kyrgyzstan" },
  { code: "LAO", name: "Laos" },
  { code: "LVA", name: "Latvia" },
  { code: "LBN", name: "Lebanon" },
  { code: "LSO", name: "Lesotho" },
  { code: "LBR", name: "Liberia" },
  { code: "LBY", name: "Libya" },
  { code: "LIE", name: "Liechtenstein" },
  { code: "LTU", name: "Lithuania" },
  { code: "LUX", name: "Luxembourg" },
  { code: "MDG", name: "Madagascar" },
  { code: "MWI", name: "Malawi" },
  { code: "MYS", name: "Malaysia" },
  { code: "MDV", name: "Maldives" },
  { code: "MLI", name: "Mali" },
  { code: "MLT", name: "Malta" },
  { code: "MHL", name: "Marshall Islands" },
  { code: "MRT", name: "Mauritania" },
  { code: "MUS", name: "Mauritius" },
  { code: "MEX", name: "Mexico" },
  { code: "FSM", name: "Micronesia" },
  { code: "MDA", name: "Moldova" },
  { code: "MCO", name: "Monaco" },
  { code: "MNG", name: "Mongolia" },
  { code: "MNE", name: "Montenegro" },
  { code: "MAR", name: "Morocco" },
  { code: "MOZ", name: "Mozambique" },
  { code: "MMR", name: "Myanmar" },
  { code: "NAM", name: "Namibia" },
  { code: "NRU", name: "Nauru" },
  { code: "NPL", name: "Nepal" },
  { code: "NLD", name: "Netherlands" },
  { code: "NZL", name: "New Zealand" },
  { code: "NIC", name: "Nicaragua" },
  { code: "NER", name: "Niger" },
  { code: "NGA", name: "Nigeria" },
  { code: "MKD", name: "North Macedonia" },
  { code: "NOR", name: "Norway" },
  { code: "OMN", name: "Oman" },
  { code: "PAK", name: "Pakistan" },
  { code: "PLW", name: "Palau" },
  { code: "PSE", name: "Palestine" },
  { code: "PAN", name: "Panama" },
  { code: "PNG", name: "Papua New Guinea" },
  { code: "PRY", name: "Paraguay" },
  { code: "PER", name: "Peru" },
  { code: "PHL", name: "Philippines" },
  { code: "POL", name: "Poland" },
  { code: "PRT", name: "Portugal" },
  { code: "QAT", name: "Qatar" },
  { code: "ROU", name: "Romania" },
  { code: "RUS", name: "Russia" },
  { code: "RWA", name: "Rwanda" },
  { code: "KNA", name: "Saint Kitts and Nevis" },
  { code: "LCA", name: "Saint Lucia" },
  { code: "VCT", name: "Saint Vincent and the Grenadines" },
  { code: "WSM", name: "Samoa" },
  { code: "SMR", name: "San Marino" },
  { code: "STP", name: "Sao Tome and Principe" },
  { code: "SAU", name: "Saudi Arabia" },
  { code: "SEN", name: "Senegal" },
  { code: "SRB", name: "Serbia" },
  { code: "SYC", name: "Seychelles" },
  { code: "SLE", name: "Sierra Leone" },
  { code: "SGP", name: "Singapore" },
  { code: "SVK", name: "Slovakia" },
  { code: "SVN", name: "Slovenia" },
  { code: "SLB", name: "Solomon Islands" },
  { code: "SOM", name: "Somalia" },
  { code: "ZAF", name: "South Africa" },
  { code: "SSD", name: "South Sudan" },
  { code: "ESP", name: "Spain" },
  { code: "LKA", name: "Sri Lanka" },
  { code: "SDN", name: "Sudan" },
  { code: "SUR", name: "Suriname" },
  { code: "SWE", name: "Sweden" },
  { code: "CHE", name: "Switzerland" },
  { code: "SYR", name: "Syria" },
  { code: "TWN", name: "Taiwan" },
  { code: "TJK", name: "Tajikistan" },
  { code: "TZA", name: "Tanzania" },
  { code: "THA", name: "Thailand" },
  { code: "TLS", name: "Timor-Leste" },
  { code: "TGO", name: "Togo" },
  { code: "TON", name: "Tonga" },
  { code: "TTO", name: "Trinidad and Tobago" },
  { code: "TUN", name: "Tunisia" },
  { code: "TUR", name: "Turkey" },
  { code: "TKM", name: "Turkmenistan" },
  { code: "TUV", name: "Tuvalu" },
  { code: "UGA", name: "Uganda" },
  { code: "UKR", name: "Latvia" },
  { code: "ARE", name: "United Arab Emirates" },
  { code: "GBR", name: "United Kingdom" },
  { code: "USA", name: "United States" },
  { code: "URY", name: "Uruguay" },
  { code: "UZB", name: "Uzbekistan" },
  { code: "VUT", name: "Vanuatu" },
  { code: "VEN", name: "Venezuela" },
  { code: "VNM", name: "Vietnam" },
  { code: "YEM", name: "Yemen" },
  { code: "ZMB", name: "Zambia" },
  { code: "ZWE", name: "Zimbabwe" },
];

export function App() {
  const [newCountry, setNewCountry] = useState("LVA");
  const [indicator, setIndicator] = useState("SP.POP.TOTL");
  const [from, setFrom] = useState<number | "">(1960);
  const [to, setTo] = useState<number | "">(2025);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [countriesData, setCountriesData] = useState<CountryData[]>([]);
  const [predictions, setPredictions] = useState<
    Record<string, Record<number, number>>
  >({});
  const [predictionDetails, setPredictionDetails] = useState<
    Record<
      string,
      {
        source?: string;
        notes?: string;
        targetYear: number;
        insights?: PredictionInsights;
        countryInsight?: CountryInsight | null;
      }
    >
  >({});
  const [predicting, setPredicting] = useState(false);
  const [predictionYears, setPredictionYears] = useState<number>(5);
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const QUICK_COUNTRY = "LVA";
  const QUICK_FROM = 1960;
  const QUICK_TO = 2025;

  // Load initial countries on mount
  useEffect(() => {
    const initialCountries = ["LVA", "EST", "LTU"]; // Latvia, Estonia, Lithuania
    const defaultIndicator = "SP.POP.TOTL";

    async function loadInitialCountries() {
      setLoading(true);
      try {
        const promises = initialCountries.map((country) =>
          fetchWorldBankData({
            country,
            indicator: defaultIndicator,
            from: 1960, // Use explicit values instead of state
            to: 2025,
          })
        );
        const results = await Promise.all(promises);
        const initialData: CountryData[] = results.map((data) => ({
          country: data.country,
          indicator: data.indicator,
          indicatorName: data.indicatorName,
          rows: data.rows
            .map((r: any) => ({ year: r.year, value: r.value }))
            .filter((r: any) => r.value !== null && r.value !== undefined), // Filter out null/undefined values
        }));
        console.log("Loaded initial countries data:", initialData);
        setCountriesData(initialData);
      } catch (err: any) {
        console.error("Error loading initial countries:", err);
        setError(err?.message || "Failed to load initial countries");
      } finally {
        setLoading(false);
      }
    }

    loadInitialCountries();
  }, []); // Only run once on mount

  const canAdd = useMemo(
    () =>
      !!newCountry &&
      !countriesData.some(
        (c) =>
          c.country === newCountry.toUpperCase() &&
          c.indicator === indicator.toUpperCase()
      ),
    [newCountry, indicator, countriesData]
  );

  async function onAddCountry(e: React.FormEvent) {
    e.preventDefault();
    if (!canAdd) return;
    const countryCode = newCountry.toUpperCase();
    const indicatorCode = indicator.toUpperCase();
    setLoading(true);
    setError(null);
    try {
      const data = await fetchWorldBankData({
        country: countryCode,
        indicator: indicatorCode,
        from: from || undefined,
        to: to || undefined,
      });
      setCountriesData([
        {
          country: countryCode,
          indicator: data.indicator,
          indicatorName: data.indicatorName,
          rows: data.rows.map((r: any) => ({ year: r.year, value: r.value })),
        },
      ]);
      setPredictions({});
      setPredictionDetails({});
    } catch (err: any) {
      setError(err?.message || "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }

  function onRemoveCountry(country: string, indicatorCode: string) {
    setCountriesData((prev) =>
      prev.filter(
        (c) => !(c.country === country && c.indicator === indicatorCode)
      )
    );
  }

  function onClearAll() {
    setCountriesData([]);
  }

  async function handleQuickIndicatorSelect(indicatorCode: string) {
    setIndicator(indicatorCode);
    setNewCountry(QUICK_COUNTRY);
    setFrom(QUICK_FROM);
    setTo(QUICK_TO);
    setLoading(true);
    setError(null);
    try {
      const data = await fetchWorldBankData({
        country: QUICK_COUNTRY,
        indicator: indicatorCode,
        from: QUICK_FROM,
        to: QUICK_TO,
      });
      setCountriesData([
        {
          country: QUICK_COUNTRY,
          indicator: data.indicator,
          indicatorName: data.indicatorName,
          rows: data.rows.map((r: any) => ({ year: r.year, value: r.value })),
        },
      ]);
      setPredictions({});
      setPredictionDetails({});
    } catch (err: any) {
      setError(err?.message || "Failed to load indicator for Latvia");
    } finally {
      setLoading(false);
    }
  }

  async function onRefreshAll() {
    if (countriesData.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const current = countriesData[0];
      const data = await fetchWorldBankData({
        country: current.country,
        indicator: current.indicator,
        from: from || undefined,
        to: to || undefined,
      });
      setCountriesData([
        {
          country: current.country,
          indicator: data.indicator,
          indicatorName: data.indicatorName,
          rows: data.rows.map((r: any) => ({ year: r.year, value: r.value })),
        },
      ]);
      setPredictions({});
      setPredictionDetails({});
    } catch (err: any) {
      setError(err?.message || "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }

  const getCountryName = (code: string) => {
    const country = COUNTRIES_LIST.find((c) => c.code === code);
    return country ? country.name : code;
  };

  async function onPredict() {
    if (countriesData.length === 0) {
      setError("Please add at least one country to predict");
      return;
    }

    setPredicting(true);
    setError(null);

    try {
      // Group countries by indicator
      const byIndicator = new Map<string, string[]>();
      for (const cd of countriesData) {
        const key = cd.indicator;
        if (!byIndicator.has(key)) {
          byIndicator.set(key, []);
        }
        byIndicator.get(key)!.push(cd.country);
      }

      const newPredictions: Record<string, Record<number, number>> = {};
      const newDetails: Record<
        string,
        {
          source?: string;
          notes?: string;
          targetYear: number;
          insights?: PredictionInsights;
          countryInsight?: CountryInsight | null;
        }
      > = {};

      const extractPredictionMap = (raw: unknown) => {
        if (!raw || typeof raw !== "object") return {};
        const candidate = raw as Record<string, unknown>;
        if (
          candidate.predictions &&
          typeof candidate.predictions === "object" &&
          !Array.isArray(candidate.predictions)
        ) {
          return candidate.predictions as Record<string, unknown>;
        }
        return candidate;
      };

      const toFiniteNumber = (value: unknown) => {
        if (typeof value === "number" && Number.isFinite(value)) return value;
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : undefined;
      };

      // Predict for each indicator group
      for (const [indicator, countries] of byIndicator.entries()) {
        try {
          // Get base year from the data
          const sampleData = countriesData.find(
            (cd) => cd.indicator === indicator
          );
          const baseYear = sampleData
            ? Math.max(...sampleData.rows.map((r) => r.year))
            : 2020;

          // Predict indicator values
          const result = await predictIndicator({
            indicator,
            countries,
            target_year: baseYear + predictionYears,
            base_year: baseYear,
          });

          const normalizedPredictions = extractPredictionMap(
            result.predictions
          );
          const resolvedTargetYear =
            toFiniteNumber(result.target_year) ?? baseYear + predictionYears;

          // Store predictions per country
          for (const [country, value] of Object.entries(
            normalizedPredictions
          )) {
            const numericValue = toFiniteNumber(value);
            if (numericValue === undefined) continue;
            const key = `${country}-${indicator}`;
            if (!newPredictions[key]) {
              newPredictions[key] = {};
            }
            newPredictions[key][resolvedTargetYear] = numericValue;

            const countryInsight =
              result.insights?.by_country?.find(
                (entry) =>
                  entry.country?.toUpperCase() === country.toUpperCase()
              ) ?? null;

            newDetails[key] = {
              source: result.source,
              notes: result.notes,
              targetYear: resolvedTargetYear,
              insights: result.insights,
              countryInsight,
            };
          }
        } catch (err: any) {
          console.error(`Prediction error for ${indicator}:`, err);
          // Continue with other indicators
        }
      }

      setPredictions(newPredictions);
      setPredictionDetails((prev) => ({
        ...prev,
        ...newDetails,
      }));
    } catch (err: any) {
      setError(err?.message || "Failed to generate predictions");
    } finally {
      setPredicting(false);
    }
  }

  function onClearPredictions() {
    setPredictions({});
  }

  return (
    <div
      style={{
        maxWidth: 1200,
        margin: "0 auto",
        padding: isMobile ? "16px 12px" : "32px 24px",
        fontFamily: "'Lato', sans-serif",
        minHeight: "100vh",
        background: "#000101",
      }}
    >
      <div style={{ marginBottom: isMobile ? "32px" : "48px" }}>
        <h1
          className={layoutStyles.heroTitle}
          style={{
            fontSize: isMobile ? "32px" : "56px",
            fontWeight: 800,
            marginBottom: isMobile ? "16px" : "24px",
            letterSpacing: "-1.5px",
            fontFamily: "'Lato', sans-serif",
            lineHeight: 1.1,
          }}
        >
          World Bank Data Explorer
        </h1>
        <p
          className={layoutStyles.heroIntro}
          style={{
            fontSize: isMobile ? "15px" : "18px",
            color: "rgba(227, 242, 253, 0.85)",
            fontWeight: 400,
            fontFamily: "'Lato', sans-serif",
            margin: 0,
          }}
        >
          Explore global development indicators with interactive visualizations
          and AI-powered predictions. Compare countries, analyze trends, and
          forecast future values using advanced machine learning models.
        </p>
      </div>
      <div className={layoutStyles.layout}>
        <aside className={layoutStyles.sidebar}>
          <div className={layoutStyles.sidebarHeader}>
            <span className={layoutStyles.sidebarTitle}>Indicators</span>
            <span className={layoutStyles.sidebarSubtitle}>
              Latvia • 1960 – 2025
            </span>
          </div>
          <p className={layoutStyles.indicatorDescription}>
            Jump straight to any indicator with Latvia's historical series
            preloaded. Clicking replaces Latvia's line while keeping any other
            countries you've added.
          </p>
          <div className={layoutStyles.rangePill}>
            <span>LVA</span>
            <span>1960–2025</span>
          </div>
          <div className={layoutStyles.indicatorList}>
            {INDICATORS.map(({ code, name }) => {
              const isActive = indicator === code;
              return (
                <button
                  type="button"
                  key={`sidebar-${code}`}
                  className={`${layoutStyles.indicatorItem} ${
                    isActive ? layoutStyles.indicatorItemActive : ""
                  }`}
                  onClick={() => handleQuickIndicatorSelect(code)}
                >
                  <span className={layoutStyles.indicatorName}>{name}</span>
                  <span className={layoutStyles.indicatorMeta}>
                    <span className={layoutStyles.indicatorCode}>{code}</span>
                    <span className={layoutStyles.indicatorAction}>
                      Load Latvia
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        </aside>

        <div className={layoutStyles.mainContent}>
          <form
            onSubmit={onAddCountry}
            className={layoutStyles.fadeInUp}
            style={{
              display: "flex",
              gap: isMobile ? 12 : 16,
              alignItems: "flex-end",
              flexWrap: "wrap",
              marginBottom: 24,
              padding: isMobile ? "16px" : "24px",
              background:
                "linear-gradient(135deg, rgba(25, 118, 210, 0.12) 0%, rgba(13, 71, 161, 0.08) 100%)",
              borderRadius: 16,
              border: "1px solid rgba(144, 202, 249, 0.15)",
              fontFamily: "'Lato', sans-serif",
              boxShadow:
                "0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(144, 202, 249, 0.1) inset",
              transition: "all 0.3s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.boxShadow =
                "0 12px 48px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(144, 202, 249, 0.2) inset";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.boxShadow =
                "0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(144, 202, 249, 0.1) inset";
            }}
          >
            <label
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                color: "#90caf9",
                fontSize: "14px",
                fontWeight: 600,
                fontFamily: "'Lato', sans-serif",
              }}
            >
              Indicator
              <select
                value={indicator}
                onChange={(e) => setIndicator(e.target.value)}
                style={{
                  minWidth: isMobile ? "100%" : 280,
                  width: isMobile ? "100%" : "auto",
                  padding: isMobile ? "12px 14px" : "10px 12px",
                  background: "rgba(25, 118, 210, 0.15)",
                  border: "1px solid rgba(144, 202, 249, 0.3)",
                  borderRadius: 8,
                  color: "#e3f2fd",
                  fontSize: "14px",
                  fontFamily: "'Lato', sans-serif",
                  cursor: "pointer",
                  outline: "none",
                  minHeight: "44px" /* Touch-friendly */,
                }}
              >
                {INDICATORS.map(({ code, name }) => {
                  // console.log(code, name);
                  return (
                    <option key={code} value={code}>
                      {name}
                    </option>
                  );
                })}
              </select>
            </label>
            <label
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                color: "#90caf9",
                fontSize: "14px",
                fontWeight: 600,
                fontFamily: "'Lato', sans-serif",
              }}
            >
              Country
              <select
                value={newCountry}
                onChange={(e) => setNewCountry(e.target.value)}
                style={{
                  minWidth: isMobile ? "100%" : 200,
                  width: isMobile ? "100%" : "auto",
                  padding: isMobile ? "12px 14px" : "10px 12px",
                  background: "rgba(25, 118, 210, 0.15)",
                  border: "1px solid rgba(144, 202, 249, 0.3)",
                  borderRadius: 8,
                  color: "#e3f2fd",
                  fontSize: "14px",
                  fontFamily: "'Lato', sans-serif",
                  cursor: "pointer",
                  outline: "none",
                  minHeight: "44px" /* Touch-friendly */,
                }}
              >
                {COUNTRIES_LIST.map(({ code, name }) => (
                  <option key={code} value={code}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                color: "#90caf9",
                fontSize: "14px",
                fontWeight: 600,
                fontFamily: "'Lato', sans-serif",
              }}
            >
              From year
              <input
                type="number"
                value={from}
                onChange={(e) =>
                  setFrom(e.target.value ? Number(e.target.value) : "")
                }
                placeholder="1960"
                style={{
                  padding: isMobile ? "12px 14px" : "10px 12px",
                  background: "rgba(25, 118, 210, 0.15)",
                  border: "1px solid rgba(144, 202, 249, 0.3)",
                  borderRadius: 8,
                  color: "#e3f2fd",
                  fontSize: "14px",
                  fontFamily: "'Lato', sans-serif",
                  width: isMobile ? "100%" : "120px",
                  outline: "none",
                  minHeight: "44px" /* Touch-friendly */,
                }}
              />
            </label>
            <label
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                color: "#90caf9",
                fontSize: "14px",
                fontWeight: 600,
                fontFamily: "'Lato', sans-serif",
              }}
            >
              To year
              <input
                type="number"
                value={to}
                onChange={(e) =>
                  setTo(e.target.value ? Number(e.target.value) : "")
                }
                placeholder="2025"
                style={{
                  padding: isMobile ? "12px 14px" : "10px 12px",
                  background: "rgba(25, 118, 210, 0.15)",
                  border: "1px solid rgba(144, 202, 249, 0.3)",
                  borderRadius: 8,
                  color: "#e3f2fd",
                  fontSize: "14px",
                  fontFamily: "'Lato', sans-serif",
                  width: isMobile ? "100%" : "120px",
                  outline: "none",
                  minHeight: "44px" /* Touch-friendly */,
                }}
              />
            </label>
            <button
              disabled={!canAdd || loading}
              type="submit"
              style={{
                padding: isMobile ? "12px 24px" : "10px 24px",
                background:
                  !canAdd || loading
                    ? "rgba(144, 202, 249, 0.2)"
                    : "linear-gradient(135deg, #1976d2 0%, #1565c0 100%)",
                color: "#e3f2fd",
                border: "none",
                borderRadius: 10,
                fontSize: "14px",
                fontWeight: 600,
                fontFamily: "'Lato', sans-serif",
                cursor: !canAdd || loading ? "not-allowed" : "pointer",
                transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                opacity: !canAdd || loading ? 0.5 : 1,
                minHeight: "44px" /* Touch-friendly */,
                width: isMobile ? "100%" : "auto",
                boxShadow:
                  !canAdd || loading
                    ? "none"
                    : "0 4px 16px rgba(25, 118, 210, 0.4)",
              }}
              onMouseEnter={(e) => {
                if (!(!canAdd || loading)) {
                  e.currentTarget.style.transform = "translateY(-2px)";
                  e.currentTarget.style.boxShadow =
                    "0 6px 24px rgba(25, 118, 210, 0.5)";
                }
              }}
              onMouseLeave={(e) => {
                if (!(!canAdd || loading)) {
                  e.currentTarget.style.transform = "translateY(0)";
                  e.currentTarget.style.boxShadow =
                    "0 4px 16px rgba(25, 118, 210, 0.4)";
                }
              }}
            >
              {loading ? "Loading..." : "Add Country"}
            </button>
            {countriesData.length > 0 && (
              <>
                <button
                  type="button"
                  onClick={onRefreshAll}
                  disabled={loading}
                  style={{
                    background: "rgba(144, 202, 249, 0.15)",
                    color: "#90caf9",
                    border: "1px solid rgba(144, 202, 249, 0.3)",
                    padding: "10px 20px",
                    borderRadius: 8,
                    cursor: loading ? "not-allowed" : "pointer",
                    fontSize: "14px",
                    fontWeight: 600,
                    fontFamily: "'Lato', sans-serif",
                    opacity: loading ? 0.5 : 1,
                    transition: "all 0.2s ease",
                  }}
                >
                  Refresh All
                </button>
                <button
                  type="button"
                  onClick={onClearAll}
                  disabled={loading}
                  style={{
                    background: "rgba(244, 67, 54, 0.2)",
                    color: "#ff8a80",
                    border: "1px solid rgba(244, 67, 54, 0.3)",
                    padding: "10px 20px",
                    borderRadius: 8,
                    cursor: loading ? "not-allowed" : "pointer",
                    fontSize: "14px",
                    fontWeight: 600,
                    fontFamily: "'Lato', sans-serif",
                    opacity: loading ? 0.5 : 1,
                    transition: "all 0.2s ease",
                  }}
                >
                  Clear All
                </button>
              </>
            )}
          </form>
          {error && (
            <p
              style={{
                color: "#ff8a80",
                marginBottom: 16,
                padding: "12px 16px",
                background: "rgba(244, 67, 54, 0.15)",
                borderRadius: 8,
                border: "1px solid rgba(244, 67, 54, 0.3)",
                fontFamily: "'Lato', sans-serif",
                fontSize: "14px",
              }}
            >
              {error}
            </p>
          )}
          {countriesData.length > 0 && (
            <>
              <div
                style={{
                  marginBottom: 24,
                  padding: "20px",
                  background: "rgba(25, 118, 210, 0.08)",
                  borderRadius: 12,
                  border: "1px solid rgba(144, 202, 249, 0.12)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 16,
                  }}
                >
                  <h3
                    style={{
                      margin: 0,
                      color: "#e3f2fd",
                      fontSize: "18px",
                      fontWeight: 600,
                      fontFamily: "'Lato', sans-serif",
                    }}
                  >
                    Selected Countries ({countriesData.length})
                  </h3>
                  <button
                    onClick={onClearAll}
                    style={{
                      background: "rgba(244, 67, 54, 0.2)",
                      color: "#ff8a80",
                      border: "1px solid rgba(244, 67, 54, 0.3)",
                      padding: "6px 14px",
                      borderRadius: 8,
                      cursor: "pointer",
                      fontSize: 13,
                      fontWeight: 600,
                      fontFamily: "'Lato', sans-serif",
                      transition: "all 0.2s ease",
                    }}
                  >
                    Clear All
                  </button>
                </div>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  {countriesData.map((cd, idx) => (
                    <div
                      key={`${cd.country}-${cd.indicator}-${idx}`}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                        padding: "10px 14px",
                        background: "rgba(144, 202, 249, 0.1)",
                        borderRadius: 8,
                        border: "1px solid rgba(144, 202, 249, 0.2)",
                      }}
                    >
                      <span
                        style={{
                          fontWeight: 600,
                          color: "#90caf9",
                          fontSize: "14px",
                          fontFamily: "'Lato', sans-serif",
                        }}
                      >
                        {getCountryName(cd.country)} - {cd.indicatorName}
                      </span>
                      <button
                        onClick={() =>
                          onRemoveCountry(cd.country, cd.indicator)
                        }
                        style={{
                          background: "rgba(244, 67, 54, 0.2)",
                          color: "#ff8a80",
                          border: "none",
                          borderRadius: 4,
                          padding: "2px 8px",
                          cursor: "pointer",
                          fontSize: 14,
                          fontFamily: "'Lato', sans-serif",
                          lineHeight: 1,
                          transition: "all 0.2s ease",
                        }}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Prediction Controls */}
              <div
                style={{
                  marginBottom: 24,
                  padding: "20px",
                  background: "rgba(25, 118, 210, 0.12)",
                  borderRadius: 12,
                  border: "1px solid rgba(144, 202, 249, 0.2)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 12,
                    flexWrap: "wrap",
                    gap: 16,
                  }}
                >
                  <h3
                    style={{
                      margin: 0,
                      color: "#90caf9",
                      fontSize: "18px",
                      fontWeight: 600,
                      fontFamily: "'Lato', sans-serif",
                    }}
                  >
                    AI Predictions
                  </h3>
                  <div
                    style={{
                      display: "flex",
                      gap: 12,
                      alignItems: "center",
                      flexWrap: "wrap",
                    }}
                  >
                    <label
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        color: "#90caf9",
                        fontSize: "14px",
                        fontFamily: "'Lato', sans-serif",
                      }}
                    >
                      <span>Years ahead:</span>
                      <input
                        type="number"
                        min="1"
                        max="20"
                        value={predictionYears}
                        onChange={(e) =>
                          setPredictionYears(Number(e.target.value))
                        }
                        style={{
                          width: 60,
                          padding: "6px 8px",
                          border: "1px solid rgba(144, 202, 249, 0.3)",
                          borderRadius: 8,
                          background: "rgba(25, 118, 210, 0.15)",
                          color: "#e3f2fd",
                          fontSize: "14px",
                          fontFamily: "'Lato', sans-serif",
                          outline: "none",
                        }}
                      />
                    </label>
                    <button
                      onClick={onPredict}
                      disabled={predicting || countriesData.length === 0}
                      style={{
                        background:
                          predicting || countriesData.length === 0
                            ? "rgba(144, 202, 249, 0.2)"
                            : "#1976d2",
                        color: "#e3f2fd",
                        border: "none",
                        padding: "10px 20px",
                        borderRadius: 8,
                        cursor:
                          predicting || countriesData.length === 0
                            ? "not-allowed"
                            : "pointer",
                        fontSize: 14,
                        fontWeight: 600,
                        fontFamily: "'Lato', sans-serif",
                        opacity:
                          predicting || countriesData.length === 0 ? 0.5 : 1,
                        transition: "all 0.2s ease",
                      }}
                    >
                      {predicting ? "Predicting..." : "Generate Predictions"}
                    </button>
                    {Object.keys(predictions).length > 0 && (
                      <button
                        onClick={onClearPredictions}
                        style={{
                          background: "rgba(144, 202, 249, 0.15)",
                          color: "#90caf9",
                          border: "1px solid rgba(144, 202, 249, 0.3)",
                          padding: "10px 20px",
                          borderRadius: 8,
                          cursor: "pointer",
                          fontSize: 14,
                          fontWeight: 600,
                          fontFamily: "'Lato', sans-serif",
                          transition: "all 0.2s ease",
                        }}
                      >
                        Clear Predictions
                      </button>
                    )}
                  </div>
                </div>
                <p
                  style={{
                    margin: 0,
                    fontSize: 13,
                    color: "#90caf9",
                    opacity: 0.8,
                    fontFamily: "'Lato', sans-serif",
                  }}
                >
                  Use AI-powered Graph Neural Network to predict future values
                  for selected countries and indicators. Predictions are shown
                  as dashed lines on the chart.
                </p>
              </div>
            </>
          )}
          <div className={layoutStyles.chartContainer}>
            <PopulationTable
              countriesData={countriesData}
              predictions={predictions}
              predictionDetails={predictionDetails}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
