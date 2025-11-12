"""
Migration data fetcher using World Bank migration indicators.
Since direct migration flow data is limited, we use:
1. Net migration rate
2. Population changes
3. Estimated flows based on economic indicators
"""
import requests
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationDataFetcher:
    """Fetches and estimates migration flows from World Bank data."""
    
    BASE_URL = "https://api.worldbank.org/v2"
    
    # Migration-related indicators
    MIGRATION_INDICATORS = {
        "net_migration": "SM.POP.NETM",  # Net migration
        "refugee_population": "SM.POP.REFG",  # Refugee population
        "refugee_by_origin": "SM.POP.REFG.OR",  # Refugees by country of origin
    }
    
    def __init__(self, cache: Optional[object] = None):
        """
        Initialize migration data fetcher.
        
        Args:
            cache: Redis cache client (optional)
        """
        self.cache = cache
    
    def fetch_indicator(
        self,
        country: str,
        indicator: str,
        year: Optional[int] = None,
    ) -> Optional[float]:
        """Fetch a single indicator value."""
        cache_key = f"wb:{indicator}:{country}:{year or 'latest'}"
        
        if self.cache:
            try:
                import json
                cached = self.cache.get(cache_key)
                if cached:
                    data = json.loads(cached) if isinstance(cached, str) else cached
                    if data and isinstance(data, dict) and "value" in data:
                        return float(data["value"])
            except Exception as e:
                logger.debug(f"Cache read error: {e}")
        
        url = f"{self.BASE_URL}/country/{country}/indicator/{indicator}"
        params = {"format": "json", "per_page": 100}
        if year:
            params["date"] = str(year)
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list) or len(data) < 2:
                return None
            
            entries = data[1]
            if not entries:
                return None
            
            # Get most recent non-null value
            for entry in reversed(entries):
                if entry.get("value") is not None:
                    value = float(entry["value"])
                    # Cache it
                    if self.cache:
                        try:
                            self.cache.setex(
                                cache_key,
                                3600,
                                json.dumps({"value": value})
                            )
                        except:
                            pass
                    return value
            
            return None
        except Exception as e:
            logger.debug(f"Error fetching {indicator} for {country}: {e}")
            return None
    
    def estimate_migration_flows(
        self,
        countries: List[str],
        year: Optional[int] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Estimate migration flows between countries.
        
        Uses multiple heuristics:
        1. Net migration rates
        2. GDP differences (economic pull factors)
        3. Population differences
        4. Geographic proximity (simplified)
        
        Args:
            countries: List of country ISO3 codes
            year: Year to estimate for
        
        Returns:
            List of (source_country, target_country, estimated_flow) tuples
        """
        flows = []
        
        # Fetch net migration for all countries
        net_migration = {}
        gdp_per_capita = {}
        population = {}
        
        for country in countries:
            net_mig = self.fetch_indicator(country, "SM.POP.NETM", year)
            if net_mig is not None:
                net_migration[country] = net_mig
            
            # Fetch GDP per capita as economic indicator
            url = f"{self.BASE_URL}/country/{country}/indicator/NY.GDP.PCAP.CD"
            params = {"format": "json", "per_page": 1}
            if year:
                params["date"] = str(year)
            try:
                resp = requests.get(url, params=params, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 1 and data[1]:
                        for entry in reversed(data[1]):
                            if entry.get("value") is not None:
                                gdp_per_capita[country] = float(entry["value"])
                                break
            except:
                pass
            
            # Fetch population
            url = f"{self.BASE_URL}/country/{country}/indicator/SP.POP.TOTL"
            params = {"format": "json", "per_page": 1}
            if year:
                params["date"] = str(year)
            try:
                resp = requests.get(url, params=params, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 1 and data[1]:
                        for entry in reversed(data[1]):
                            if entry.get("value") is not None:
                                population[country] = float(entry["value"])
                                break
            except:
                pass
        
        # Estimate flows based on heuristics
        for src in countries:
            for tgt in countries:
                if src == tgt:
                    continue
                
                # Skip if we don't have data
                if src not in net_migration or tgt not in net_migration:
                    continue
                
                # Heuristic 1: Net migration indicates direction
                # Positive net migration = destination, negative = source
                src_net = net_migration.get(src, 0)
                tgt_net = net_migration.get(tgt, 0)
                
                # Flow from source to target if:
                # - Source has negative net migration (losing population)
                # - Target has positive net migration (gaining population)
                if src_net < 0 and tgt_net > 0:
                    # Estimate flow based on:
                    # 1. Magnitude of net migration
                    # 2. GDP difference (economic pull)
                    # 3. Population sizes
                    
                    base_flow = abs(src_net) * 0.1  # Scale down
                    
                    # Economic factor: higher GDP attracts more migrants
                    gdp_factor = 1.0
                    if src in gdp_per_capita and tgt in gdp_per_capita:
                        gdp_ratio = gdp_per_capita[tgt] / max(gdp_per_capita[src], 1)
                        gdp_factor = min(gdp_ratio / 2.0, 2.0)  # Cap at 2x
                    
                    # Population factor: larger countries have more migrants
                    pop_factor = 1.0
                    if src in population:
                        # Normalize by 1M population
                        pop_factor = min(population[src] / 1_000_000, 10.0)
                    
                    estimated_flow = base_flow * gdp_factor * pop_factor
                    
                    if estimated_flow > 100:  # Only include significant flows
                        flows.append((src, tgt, estimated_flow))
        
        logger.info(f"Estimated {len(flows)} migration flows for {len(countries)} countries")
        return flows
    
    def get_historical_migration_data(
        self,
        countries: List[str],
        years: List[int],
    ) -> Dict[int, List[Tuple[str, str, float]]]:
        """
        Get historical migration flows for multiple years.
        
        Args:
            countries: List of country ISO3 codes
            years: List of years
        
        Returns:
            Dictionary mapping year to list of (source, target, flow) tuples
        """
        historical_data = {}
        
        for year in years:
            flows = self.estimate_migration_flows(countries, year)
            historical_data[year] = flows
        
        return historical_data

