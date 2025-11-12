import { spawn } from "child_process";
import path from "path";

const PYTHON_EXECUTABLE = process.env.PYTHON_EXECUTABLE || "python3";
const SCRIPT_PATH = process.env.ML_SCRIPT_PATH
  ? path.resolve(process.env.ML_SCRIPT_PATH)
  : path.resolve(process.cwd(), "..", "ml-service", "src", "run_prediction.py");
const CACHE_TTL_MS = Number(process.env.ML_CACHE_TTL_MS || 5 * 60 * 1000);

interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

const cache = new Map<string, CacheEntry<any>>();

function getCache<T>(key: string): T | null {
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() > entry.expiresAt) {
    cache.delete(key);
    return null;
  }
  return entry.value as T;
}

function setCache<T>(key: string, value: T): void {
  cache.set(key, { value, expiresAt: Date.now() + CACHE_TTL_MS });
}

async function runPython(args: string[], timeoutMs = 120000): Promise<any> {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_EXECUTABLE, [SCRIPT_PATH, ...args], {
      cwd: path.resolve(process.cwd(), ".."),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
      },
    });

    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("ML script timeout"));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (err) => {
      clearTimeout(timeout);
      reject(err);
    });

    child.on("close", (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        reject(new Error(stderr || `ML script exited with code ${code}`));
        return;
      }
      try {
        const trimmed = stdout.trim();
        resolve(trimmed ? JSON.parse(trimmed) : null);
      } catch (err) {
        reject(err);
      }
    });
  });
}

export async function getIndicatorPredictionOnDemand(params: {
  indicator: string;
  countries: string[];
  targetYear?: number;
  baseYear?: number;
}): Promise<any | null> {
  const { indicator, countries, targetYear, baseYear } = params;
  const key = JSON.stringify({
    mode: "indicator",
    indicator,
    countries: [...countries].sort(),
    targetYear,
    baseYear,
  });

  const cached = getCache<any>(key);
  if (cached) {
    return cached;
  }

  const args: string[] = ["--mode", "indicator", "--indicator", indicator, "--countries", countries.join(",")];
  if (typeof targetYear === "number") {
    args.push("--target-year", String(targetYear));
  }
  if (typeof baseYear === "number") {
    args.push("--base-year", String(baseYear));
  }

  const result = await runPython(args);
  if (result) {
    setCache(key, result);
  }
  return result;
}

export async function getPopulationPredictionOnDemand(params: {
  country: string;
  yearsAhead?: number;
  baseYear?: number;
}): Promise<any | null> {
  const { country, yearsAhead, baseYear } = params;
  const key = JSON.stringify({
    mode: "population",
    country,
    yearsAhead,
    baseYear,
  });
  const cached = getCache<any>(key);
  if (cached) {
    return cached;
  }

  const args: string[] = ["--mode", "population", "--country", country];
  if (typeof yearsAhead === "number") {
    args.push("--years-ahead", String(yearsAhead));
  }
  if (typeof baseYear === "number") {
    args.push("--base-year", String(baseYear));
  }

  const result = await runPython(args);
  if (result) {
    setCache(key, result);
  }
  return result;
}
