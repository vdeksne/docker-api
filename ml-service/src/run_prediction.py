import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from indicator_predictor import IndicatorPredictor  # type: ignore
    from predictor import MigrationPredictor  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"Failed to import predictors: {exc}") from exc


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:  # pragma: no cover
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _ensure_indicator_payload(raw: Dict[str, Any], indicator: str, countries: List[str], target_year: int | None, base_year: int | None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "indicator": indicator,
        "countries": countries,
    }
    if target_year is not None:
        payload["target_year"] = target_year
    if base_year is not None:
        payload["base_year"] = base_year

    if not isinstance(raw, dict):
        payload["predictions"] = raw
        return payload

    # The underlying predictor sometimes returns the fully structured payload already
    if "predictions" in raw:
        payload.update(raw)
        return payload

    payload["predictions"] = raw
    return payload


def _ensure_population_payload(raw: Dict[int, float] | Dict[str, Any], country: str, base_year: int | None, years_ahead: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "country": country,
        "years_ahead": years_ahead,
    }
    if base_year is not None:
        payload["base_year"] = base_year

    if isinstance(raw, dict) and "predictions" in raw:
        payload.update(raw)
        return payload

    payload["predictions"] = raw
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="On-demand ML prediction runner")
    parser.add_argument("--mode", choices=["indicator", "population"], required=True)
    parser.add_argument("--indicator")
    parser.add_argument("--countries")
    parser.add_argument("--country")
    parser.add_argument("--target-year", type=int, dest="target_year")
    parser.add_argument("--base-year", type=int, dest="base_year")
    parser.add_argument("--years-ahead", type=int, dest="years_ahead")
    args = parser.parse_args()

    model_path = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "../models/migration_gnn.pth"))
    device = os.getenv("DEVICE", "cpu")

    if args.mode == "indicator":
        if not args.indicator or not args.countries:
            parser.error("--indicator and --countries are required for indicator mode")
        countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
        indicator = args.indicator.upper()
        predictor = IndicatorPredictor(model_path=model_path, device=device, cache=None)
        result = predictor.predict_indicator(
            indicator=indicator,
            countries=countries,
            target_year=args.target_year,
            base_year=args.base_year,
        )
        payload = _ensure_indicator_payload(result, indicator, countries, args.target_year, args.base_year)
    else:
        if not args.country:
            parser.error("--country is required for population mode")
        years_ahead = args.years_ahead if args.years_ahead and args.years_ahead > 0 else 5
        country = args.country.upper()
        predictor = MigrationPredictor(model_path=model_path, device=device, cache=None)
        result = predictor.predict_population_growth(
            country=country,
            years_ahead=years_ahead,
            base_year=args.base_year,
        )
        payload = _ensure_population_payload(result, country, args.base_year, years_ahead)

    json.dump(payload, sys.stdout, default=_json_default)
    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(json.dumps({"error": str(exc)}) + "\n")
        sys.exit(1)
