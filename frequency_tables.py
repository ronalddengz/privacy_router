"""
frequency_tables.py - Population frequency data from authoritative sources

Fetches and caches data from:
- U.S. Census Bureau American Community Survey (ACS) PUMS
- Bureau of Labor Statistics (BLS) Occupational Employment Statistics
- Orphanet rare disease prevalence data
- CDC health statistics

Data is cached locally in SQLite for:
1. Predictable latency (no live API calls during routing)
2. Privacy (query values not leaked to external services)
3. Joint distribution estimation via Fréchet bounds
"""

import sqlite3
import json
import math
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


@dataclass
class FrequencyResult:
    """Result of a frequency lookup."""
    count: Optional[int] = None
    frequency: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    source: str = ""
    vintage: str = ""
    is_available: bool = False
    is_exact_match: bool = False
    population_size: Optional[int] = None


@dataclass
class DataSourceInfo:
    """Metadata about a data source."""
    name: str
    url: str
    vintage: str
    last_updated: Optional[datetime] = None
    record_count: int = 0


class CensusAPIClient:
    """
    Client for U.S. Census Bureau API.
    
    Data sources:
    - ACS 5-Year PUMS: Age, sex, race, geography, marital status, citizenship
    - ACS detailed tables: Cross-tabulations by geography
    
    API documentation: https://www.census.gov/data/developers/data-sets.html
    """
    
    BASE_URL = "https://api.census.gov/data"
    
    # ACS 5-year estimates - most recent available
    ACS_YEAR = "2022"
    ACS_DATASET = "acs/acs5"
    
    # PUMS (Public Use Microdata Sample) for individual-level estimates
    PUMS_DATASET = "acs/acs5/pums"
    
    # Variable codes for ACS
    VARIABLES = {
        'total_pop': 'B01001_001E',  # Total population
        'male': 'B01001_002E',        # Male
        'female': 'B01001_026E',      # Female
        # Age by sex (male)
        'male_under_5': 'B01001_003E',
        'male_5_9': 'B01001_004E',
        'male_10_14': 'B01001_005E',
        'male_15_17': 'B01001_006E',
        'male_18_19': 'B01001_007E',
        'male_20': 'B01001_008E',
        'male_21': 'B01001_009E',
        'male_22_24': 'B01001_010E',
        'male_25_29': 'B01001_011E',
        'male_30_34': 'B01001_012E',
        'male_35_39': 'B01001_013E',
        'male_40_44': 'B01001_014E',
        'male_45_49': 'B01001_015E',
        'male_50_54': 'B01001_016E',
        'male_55_59': 'B01001_017E',
        'male_60_61': 'B01001_018E',
        'male_62_64': 'B01001_019E',
        'male_65_66': 'B01001_020E',
        'male_67_69': 'B01001_021E',
        'male_70_74': 'B01001_022E',
        'male_75_79': 'B01001_023E',
        'male_80_84': 'B01001_024E',
        'male_85_plus': 'B01001_025E',
        # Age by sex (female) - similar pattern B01001_027E to B01001_049E
        # Race
        'white_alone': 'B02001_002E',
        'black_alone': 'B02001_003E',
        'aian_alone': 'B02001_004E',  # American Indian/Alaska Native
        'asian_alone': 'B02001_005E',
        'nhpi_alone': 'B02001_006E',  # Native Hawaiian/Pacific Islander
        'other_alone': 'B02001_007E',
        'two_or_more': 'B02001_008E',
        # Hispanic origin
        'hispanic': 'B03003_003E',
        'not_hispanic': 'B03003_002E',
        # Citizenship
        'native_citizen': 'B05001_002E',
        'naturalized': 'B05001_005E',
        'not_citizen': 'B05001_006E',
        # Marital status (15+)
        'never_married': 'B12001_003E',
        'now_married': 'B12001_004E',
        'separated': 'B12001_009E',
        'widowed': 'B12001_005E',
        'divorced': 'B12001_010E',
    }
    
    # State FIPS codes
    STATE_FIPS = {
        'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
        'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12',
        'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
        'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
        'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
        'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
        'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
        'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
        'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
        'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
        'WY': '56', 'PR': '72',
    }
    
    FIPS_TO_STATE = {v: k for k, v in STATE_FIPS.items()}
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Census API client.
        
        Args:
            api_key: Census API key (get free at https://api.census.gov/data/key_signup.html)
                     Optional but recommended for higher rate limits.
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PrivacyRouter/1.0 (research use)'
        })
    
    def _make_request(self, url: str, params: dict) -> Optional[list]:
        """Make API request with retry logic."""
        if self.api_key:
            params['key'] = self.api_key
        
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Census API request failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None
    
    def get_national_demographics(self) -> dict:
        """
        Fetch national-level demographic marginals.
        
        Returns dict with counts for age buckets, sex, race, etc.
        """
        url = f"{self.BASE_URL}/{self.ACS_YEAR}/{self.ACS_DATASET}"
        
        # Fetch basic demographics
        variables = ','.join([
            'NAME',
            self.VARIABLES['total_pop'],
            self.VARIABLES['male'],
            self.VARIABLES['female'],
            self.VARIABLES['white_alone'],
            self.VARIABLES['black_alone'],
            self.VARIABLES['asian_alone'],
            self.VARIABLES['aian_alone'],
            self.VARIABLES['nhpi_alone'],
            self.VARIABLES['two_or_more'],
            self.VARIABLES['hispanic'],
            self.VARIABLES['native_citizen'],
            self.VARIABLES['naturalized'],
            self.VARIABLES['not_citizen'],
        ])
        
        params = {
            'get': variables,
            'for': 'us:1',
        }
        
        data = self._make_request(url, params)
        if not data or len(data) < 2:
            return {}
        
        headers = data[0]
        values = data[1]
        result = dict(zip(headers, values))
        
        return result
    
    def get_state_populations(self) -> dict[str, int]:
        """Fetch population by state."""
        url = f"{self.BASE_URL}/{self.ACS_YEAR}/{self.ACS_DATASET}"
        
        params = {
            'get': f"NAME,{self.VARIABLES['total_pop']}",
            'for': 'state:*',
        }
        
        data = self._make_request(url, params)
        if not data:
            return {}
        
        state_pops = {}
        for row in data[1:]:  # Skip header
            state_fips = row[2]
            state_abbr = self.FIPS_TO_STATE.get(state_fips)
            if state_abbr:
                try:
                    state_pops[state_abbr] = int(row[1])
                except (ValueError, TypeError):
                    pass
        
        return state_pops
    
    def get_age_sex_distribution(self) -> list[dict]:
        """
        Fetch age by sex distribution from ACS.
        
        Returns list of dicts with age_bucket, sex, count, frequency.
        """
        url = f"{self.BASE_URL}/{self.ACS_YEAR}/{self.ACS_DATASET}"
        
        # Build variable list for all age groups
        age_vars_male = [
            ('0-4', 'B01001_003E'),
            ('5-9', 'B01001_004E'),
            ('10-14', 'B01001_005E'),
            ('15-19', 'B01001_006E,B01001_007E'),  # 15-17 + 18-19
            ('20-24', 'B01001_008E,B01001_009E,B01001_010E'),  # 20 + 21 + 22-24
            ('25-29', 'B01001_011E'),
            ('30-34', 'B01001_012E'),
            ('35-39', 'B01001_013E'),
            ('40-44', 'B01001_014E'),
            ('45-49', 'B01001_015E'),
            ('50-54', 'B01001_016E'),
            ('55-59', 'B01001_017E'),
            ('60-64', 'B01001_018E,B01001_019E'),  # 60-61 + 62-64
            ('65-69', 'B01001_020E,B01001_021E'),  # 65-66 + 67-69
            ('70-74', 'B01001_022E'),
            ('75-79', 'B01001_023E'),
            ('80-84', 'B01001_024E'),
            ('85+', 'B01001_025E'),
        ]
        
        age_vars_female = [
            ('0-4', 'B01001_027E'),
            ('5-9', 'B01001_028E'),
            ('10-14', 'B01001_029E'),
            ('15-19', 'B01001_030E,B01001_031E'),
            ('20-24', 'B01001_032E,B01001_033E,B01001_034E'),
            ('25-29', 'B01001_035E'),
            ('30-34', 'B01001_036E'),
            ('35-39', 'B01001_037E'),
            ('40-44', 'B01001_038E'),
            ('45-49', 'B01001_039E'),
            ('50-54', 'B01001_040E'),
            ('55-59', 'B01001_041E'),
            ('60-64', 'B01001_042E,B01001_043E'),
            ('65-69', 'B01001_044E,B01001_045E'),
            ('70-74', 'B01001_046E'),
            ('75-79', 'B01001_047E'),
            ('80-84', 'B01001_048E'),
            ('85+', 'B01001_049E'),
        ]
        
        # Collect all unique variables
        all_vars = set()
        for _, vars_str in age_vars_male + age_vars_female:
            all_vars.update(vars_str.split(','))
        all_vars.add(self.VARIABLES['total_pop'])
        
        params = {
            'get': ','.join(sorted(all_vars)),
            'for': 'us:1',
        }
        
        data = self._make_request(url, params)
        if not data or len(data) < 2:
            return []
        
        headers = data[0]
        values = data[1]
        counts = {h: int(v) if v and v != 'null' else 0 for h, v in zip(headers, values)}
        
        total_pop = counts.get(self.VARIABLES['total_pop'], 1)
        results = []
        
        # Process male age buckets
        for age_bucket, vars_str in age_vars_male:
            var_list = vars_str.split(',')
            count = sum(counts.get(v, 0) for v in var_list)
            # Convert to 5-year buckets for consistency
            bucket_5yr = self._convert_to_5yr_bucket(age_bucket)
            results.append({
                'age_bucket': bucket_5yr,
                'sex': 'male',
                'count': count,
                'frequency': count / total_pop if total_pop > 0 else 0,
            })
        
        # Process female age buckets
        for age_bucket, vars_str in age_vars_female:
            var_list = vars_str.split(',')
            count = sum(counts.get(v, 0) for v in var_list)
            bucket_5yr = self._convert_to_5yr_bucket(age_bucket)
            results.append({
                'age_bucket': bucket_5yr,
                'sex': 'female',
                'count': count,
                'frequency': count / total_pop if total_pop > 0 else 0,
            })
        
        return results
    
    def _convert_to_5yr_bucket(self, bucket: str) -> str:
        """Convert various age bucket formats to 5-year buckets."""
        # Handle special cases
        if bucket == '85+':
            return '85-89'  # Or could use '85+'
        
        # Parse start age
        if '-' in bucket:
            start = int(bucket.split('-')[0])
        else:
            start = int(bucket.replace('+', ''))
        
        # Round down to nearest 5
        bucket_start = (start // 5) * 5
        bucket_end = bucket_start + 4
        
        return f"{bucket_start}-{bucket_end}"
    
    def get_marital_status_distribution(self) -> list[dict]:
        """Fetch marital status distribution."""
        url = f"{self.BASE_URL}/{self.ACS_YEAR}/{self.ACS_DATASET}"
        
        params = {
            'get': ','.join([
                self.VARIABLES['total_pop'],
                'B12001_001E',  # Total 15+
                'B12001_003E',  # Never married male
                'B12001_012E',  # Never married female
                'B12001_004E',  # Now married male
                'B12001_013E',  # Now married female
                'B12001_005E',  # Widowed male
                'B12001_014E',  # Widowed female
                'B12001_010E',  # Divorced male
                'B12001_019E',  # Divorced female
            ]),
            'for': 'us:1',
        }
        
        data = self._make_request(url, params)
        if not data or len(data) < 2:
            return []
        
        headers = data[0]
        values = data[1]
        counts = {h: int(v) if v and v != 'null' else 0 for h, v in zip(headers, values)}
        
        total_15plus = counts.get('B12001_001E', 1)
        
        return [
            {'status': 'single', 'count': counts.get('B12001_003E', 0) + counts.get('B12001_012E', 0),
             'frequency': (counts.get('B12001_003E', 0) + counts.get('B12001_012E', 0)) / total_15plus},
            {'status': 'married', 'count': counts.get('B12001_004E', 0) + counts.get('B12001_013E', 0),
             'frequency': (counts.get('B12001_004E', 0) + counts.get('B12001_013E', 0)) / total_15plus},
            {'status': 'widowed', 'count': counts.get('B12001_005E', 0) + counts.get('B12001_014E', 0),
             'frequency': (counts.get('B12001_005E', 0) + counts.get('B12001_014E', 0)) / total_15plus},
            {'status': 'divorced', 'count': counts.get('B12001_010E', 0) + counts.get('B12001_019E', 0),
             'frequency': (counts.get('B12001_010E', 0) + counts.get('B12001_019E', 0)) / total_15plus},
        ]


class BLSAPIClient:
    """
    Client for Bureau of Labor Statistics API.
    
    Data source: Occupational Employment and Wage Statistics (OEWS)
    API documentation: https://www.bls.gov/developers/
    """
    
    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    
    # Major occupation groups from Standard Occupational Classification (SOC)
    # https://www.bls.gov/soc/2018/major_groups.htm
    OCCUPATION_CODES = {
        'management': '11-0000',
        'business_financial': '13-0000',
        'computer_mathematical': '15-0000',
        'architecture_engineering': '17-0000',
        'life_physical_social_science': '19-0000',
        'community_social_service': '21-0000',
        'legal': '23-0000',
        'education': '25-0000',
        'arts_entertainment': '27-0000',
        'healthcare_practitioners': '29-0000',
        'healthcare_support': '31-0000',
        'protective_service': '33-0000',
        'food_preparation': '35-0000',
        'building_maintenance': '37-0000',
        'personal_care': '39-0000',
        'sales': '41-0000',
        'office_administrative': '43-0000',
        'farming_fishing': '45-0000',
        'construction': '47-0000',
        'installation_maintenance': '49-0000',
        'production': '51-0000',
        'transportation': '53-0000',
    }
    
    # Specific rare occupations with SOC codes
    RARE_OCCUPATIONS = {
        'zoologist': '19-1023',  # Zoologists and Wildlife Biologists
        'astronomer': '19-2011',  # Astronomers
        'epidemiologist': '19-1041',  # Epidemiologists
        'geographer': '19-3092',  # Geographers
        'historian': '19-3093',  # Historians
        'anthropologist': '19-3091',  # Anthropologists and Archeologists
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize BLS API client.
        
        Args:
            api_key: BLS API key (register at https://data.bls.gov/registrationEngine/)
                     Required for more than 25 queries/day.
        """
        self.api_key = api_key
        self.session = requests.Session()
    
    def get_occupation_employment(self) -> dict[str, dict]:
        """
        Fetch employment counts by occupation group.
        
        Returns dict mapping occupation category to employment stats.
        """
        # BLS OEWS series IDs follow pattern: OEUM[area][industry][occupation][datatype]
        # National, all industries: OEUN000000000000[SOC][01] for employment
        
        series_ids = []
        occ_map = {}
        
        for occ_name, soc_code in self.OCCUPATION_CODES.items():
            # National employment series
            series_id = f"OEUN00000000000{soc_code.replace('-', '')}01"
            series_ids.append(series_id)
            occ_map[series_id] = occ_name
        
        # Add rare occupations
        for occ_name, soc_code in self.RARE_OCCUPATIONS.items():
            series_id = f"OEUN00000000000{soc_code.replace('-', '')}01"
            series_ids.append(series_id)
            occ_map[series_id] = occ_name
        
        # BLS API accepts up to 50 series per request
        results = {}
        
        for i in range(0, len(series_ids), 50):
            batch = series_ids[i:i+50]
            batch_results = self._fetch_series(batch)
            
            for series_id, value in batch_results.items():
                occ_name = occ_map.get(series_id)
                if occ_name and value:
                    results[occ_name] = {
                        'employment': value,
                        'soc_code': self.OCCUPATION_CODES.get(occ_name) or self.RARE_OCCUPATIONS.get(occ_name),
                    }
        
        return results
    
    def _fetch_series(self, series_ids: list[str]) -> dict[str, int]:
        """Fetch data for a batch of series IDs."""
        payload = {
            'seriesid': series_ids,
            'startyear': '2023',
            'endyear': '2023',
        }
        
        if self.api_key:
            payload['registrationkey'] = self.api_key
        
        try:
            response = self.session.post(
                self.BASE_URL,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = {}
            if data.get('status') == 'REQUEST_SUCCEEDED':
                for series in data.get('Results', {}).get('series', []):
                    series_id = series.get('seriesID')
                    values = series.get('data', [])
                    if values:
                        # Get most recent value
                        try:
                            results[series_id] = int(float(values[0].get('value', 0)) * 1000)  # BLS reports in thousands
                        except (ValueError, TypeError):
                            pass
            
            return results
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"BLS API request failed: {e}")
            return {}


class OrphanetClient:
    """
    Client for Orphanet rare disease data.
    
    Data source: Orphanet (https://www.orpha.net)
    Uses the Orphadata XML/JSON exports for prevalence data.
    
    Note: Orphanet data requires acceptance of terms of use.
    Download from: https://www.orphadata.com/epidemiology/
    """
    
    # Orphanet prevalence categories
    PREVALENCE_CATEGORIES = {
        '>1 / 1,000': 0.001,
        '1-5 / 10,000': 0.00025,  # Midpoint
        '1-9 / 100,000': 0.00005,  # Midpoint
        '1-9 / 1,000,000': 0.000005,  # Midpoint
        '<1 / 1,000,000': 0.0000005,
        'Unknown': None,
        'Not yet documented': None,
    }
    
    # Selected rare diseases with known prevalence (from Orphanet)
    # These are manually curated for the conditions we detect
    RARE_DISEASE_PREVALENCE = {
        # Ehlers-Danlos syndromes (all types combined: ~1/5000)
        'ehlers_danlos': {
            'orpha_code': 'ORPHA:98249',
            'prevalence_per_100k': 20.0,  # ~1/5000
            'prevalence_category': '1-5 / 10,000',
        },
        # Huntington disease (~5-10/100,000)
        'huntingtons': {
            'orpha_code': 'ORPHA:399',
            'prevalence_per_100k': 7.5,
            'prevalence_category': '1-9 / 100,000',
        },
        # Marfan syndrome (~1-2/10,000)
        'marfan': {
            'orpha_code': 'ORPHA:558',
            'prevalence_per_100k': 15.0,
            'prevalence_category': '1-5 / 10,000',
        },
        # Cystic fibrosis (~1/3500 in Caucasians, varies by ethnicity)
        'cystic_fibrosis': {
            'orpha_code': 'ORPHA:586',
            'prevalence_per_100k': 28.6,  # ~1/3500
            'prevalence_category': '1-5 / 10,000',
        },
        # Phenylketonuria (~1/10,000)
        'pku': {
            'orpha_code': 'ORPHA:716',
            'prevalence_per_100k': 10.0,
            'prevalence_category': '1-9 / 100,000',
        },
        # Amyotrophic lateral sclerosis (~5/100,000)
        'als': {
            'orpha_code': 'ORPHA:803',
            'prevalence_per_100k': 5.0,
            'prevalence_category': '1-9 / 100,000',
        },
        # Sickle cell disease (~1/500 in African Americans)
        'sickle_cell': {
            'orpha_code': 'ORPHA:232',
            'prevalence_per_100k': 30.0,  # US average
            'prevalence_category': '1-5 / 10,000',
        },
        # Hemophilia A (~1/5000 males)
        'hemophilia_a': {
            'orpha_code': 'ORPHA:98878',
            'prevalence_per_100k': 10.0,  # Both sexes
            'prevalence_category': '1-9 / 100,000',
        },
    }
    
    # Orphadata API endpoint (if using API instead of bulk download)
    API_URL = "https://api.orphadata.com/rd-api/v1"
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize Orphanet client.
        
        Args:
            data_dir: Directory containing downloaded Orphadata files.
                      If None, uses built-in prevalence data.
        """
        self.data_dir = data_dir
        self.session = requests.Session()
    
    def get_disease_prevalence(self, disease_key: str) -> Optional[dict]:
        """Get prevalence data for a specific disease."""
        return self.RARE_DISEASE_PREVALENCE.get(disease_key)
    
    def get_all_rare_diseases(self) -> dict[str, dict]:
        """Get all known rare disease prevalence data."""
        return self.RARE_DISEASE_PREVALENCE.copy()
    
    def search_disease(self, term: str) -> list[dict]:
        """
        Search Orphanet for disease by name.
        
        Note: Requires API access or local data file.
        """
        # If API access is available
        try:
            response = self.session.get(
                f"{self.API_URL}/diseases",
                params={'name': term},
                timeout=10
            )
            if response.ok:
                return response.json().get('data', [])
        except requests.exceptions.RequestException:
            pass
        
        # Fall back to local search
        results = []
        term_lower = term.lower().replace('-', '_').replace(' ', '_')
        
        for key, data in self.RARE_DISEASE_PREVALENCE.items():
            if term_lower in key or key in term_lower:
                results.append({
                    'disease_key': key,
                    **data
                })
        
        return results


class CDCDataClient:
    """
    Client for CDC health statistics data.
    
    Data sources:
    - NHANES (National Health and Nutrition Examination Survey)
    - BRFSS (Behavioral Risk Factor Surveillance System)
    - NVSS (National Vital Statistics System)
    
    API: https://data.cdc.gov/
    """
    
    # CDC SODA API endpoint
    BASE_URL = "https://data.cdc.gov/resource"
    
    # Dataset IDs for common conditions
    DATASETS = {
        'diabetes_prevalence': 'f5xn-7w3c',  # Diabetes surveillance
        'heart_disease_mortality': 'bi63-dtpu',
        'chronic_conditions': '9dzk-mvmi',
    }
    
    # Common chronic condition prevalence (from CDC published statistics)
    CONDITION_PREVALENCE = {
        # Highly prevalent conditions
        'hypertension': {
            'prevalence_pct': 47.0,  # Adults 18+
            'source': 'NHANES 2017-2020',
            'age_group': '18+',
        },
        'hyperlipidemia': {
            'prevalence_pct': 38.0,  # High cholesterol
            'source': 'NHANES',
            'age_group': '20+',
        },
        'obesity': {
            'prevalence_pct': 41.9,
            'source': 'NHANES 2017-2020',
            'age_group': '20+',
        },
        'diabetes_type2': {
            'prevalence_pct': 11.3,  # Diagnosed + undiagnosed
            'source': 'CDC National Diabetes Statistics Report 2022',
            'age_group': '18+',
        },
        'arthritis': {
            'prevalence_pct': 23.7,
            'source': 'NHIS 2019',
            'age_group': '18+',
        },
        'asthma': {
            'prevalence_pct': 7.7,
            'source': 'CDC 2021',
            'age_group': 'all',
        },
        'depression': {
            'prevalence_pct': 18.4,
            'source': 'NHIS 2019',
            'age_group': '18+',
        },
        'anxiety': {
            'prevalence_pct': 15.6,
            'source': 'NHIS 2019',
            'age_group': '18+',
        },
        'copd': {
            'prevalence_pct': 4.5,
            'source': 'NHIS 2020',
            'age_group': '18+',
        },
        'cancer': {
            'prevalence_pct': 5.8,  # History of cancer
            'source': 'NHIS 2019',
            'age_group': '18+',
        },
        'chronic_kidney_disease': {
            'prevalence_pct': 14.0,
            'source': 'CDC CKD Surveillance',
            'age_group': '18+',
        },
    }
    
    def __init__(self, app_token: Optional[str] = None):
        """
        Initialize CDC data client.
        
        Args:
            app_token: Socrata app token for higher rate limits.
        """
        self.app_token = app_token
        self.session = requests.Session()
        if app_token:
            self.session.headers['X-App-Token'] = app_token
    
    def get_condition_prevalence(self, condition: str) -> Optional[dict]:
        """Get prevalence data for a condition."""
        return self.CONDITION_PREVALENCE.get(condition)
    
    def get_all_conditions(self) -> dict[str, dict]:
        """Get all condition prevalence data."""
        return self.CONDITION_PREVALENCE.copy()


class LocalFrequencyTable:
    """
    Local SQLite-based frequency tables for joint QI combinations.
    
    Data is populated from authoritative sources:
    - Census Bureau ACS (demographics, geography)
    - Bureau of Labor Statistics (occupations)
    - Orphanet (rare disease prevalence)
    - CDC (common condition prevalence)
    
    Tables are refreshed offline and versioned for reproducibility.
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS population_info (
        population_id TEXT PRIMARY KEY,
        name TEXT,
        vintage TEXT,
        total_size INTEGER,
        description TEXT,
        created_at TEXT,
        sources TEXT  -- JSON array of data sources used
    );
    
    CREATE TABLE IF NOT EXISTS joint_frequencies (
        population_id TEXT,
        qi_combination TEXT,  -- JSON array of QI types
        qi_values TEXT,       -- JSON object of normalized values
        count INTEGER,
        frequency REAL,
        lower_bound REAL,
        upper_bound REAL,
        source TEXT,          -- Data source for this joint frequency
        FOREIGN KEY (population_id) REFERENCES population_info(population_id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_joint_lookup 
    ON joint_frequencies(population_id, qi_combination, qi_values);
    
    CREATE TABLE IF NOT EXISTS marginal_frequencies (
        population_id TEXT,
        qi_type TEXT,
        qi_value TEXT,
        count INTEGER,
        frequency REAL,
        lower_bound REAL,
        upper_bound REAL,
        source TEXT,          -- Specific data source (e.g., 'census_acs_2022')
        vintage TEXT,
        FOREIGN KEY (population_id) REFERENCES population_info(population_id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_marginal_lookup 
    ON marginal_frequencies(population_id, qi_type, qi_value);
    
    CREATE TABLE IF NOT EXISTS data_source_metadata (
        source_id TEXT PRIMARY KEY,
        name TEXT,
        url TEXT,
        vintage TEXT,
        fetch_date TEXT,
        record_count INTEGER,
        checksum TEXT
    );
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)
    
    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def get_population_info(self, population_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM population_info WHERE population_id = ?",
                (population_id,)
            ).fetchone()
            return dict(row) if row else None
    
    def lookup_joint(
        self,
        population_id: str,
        qi_values: dict[str, str]
    ) -> FrequencyResult:
        """
        Look up exact joint frequency for a combination of QI values.
        """
        qi_combination = json.dumps(sorted(qi_values.keys()))
        qi_values_json = json.dumps(qi_values, sort_keys=True)
        
        with self._connect() as conn:
            row = conn.execute("""
                SELECT count, frequency, lower_bound, upper_bound, source
                FROM joint_frequencies 
                WHERE population_id = ? 
                  AND qi_combination = ?
                  AND qi_values = ?
            """, (population_id, qi_combination, qi_values_json)).fetchone()
            
            if row:
                pop_info = self.get_population_info(population_id)
                return FrequencyResult(
                    count=row['count'],
                    frequency=row['frequency'],
                    lower_bound=row['lower_bound'],
                    upper_bound=row['upper_bound'],
                    source=row['source'] or population_id,
                    vintage=pop_info.get('vintage', '') if pop_info else '',
                    is_available=True,
                    is_exact_match=True,
                    population_size=pop_info.get('total_size') if pop_info else None
                )
        
        return FrequencyResult(is_available=False)
    
    def lookup_marginal(
        self,
        population_id: str,
        qi_type: str,
        qi_value: str
    ) -> FrequencyResult:
        """Look up a single marginal frequency."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT count, frequency, lower_bound, upper_bound, source, vintage
                FROM marginal_frequencies
                WHERE population_id = ? AND qi_type = ? AND qi_value = ?
            """, (population_id, qi_type, qi_value)).fetchone()
            
            if row:
                pop_info = self.get_population_info(population_id)
                return FrequencyResult(
                    count=row['count'],
                    frequency=row['frequency'],
                    lower_bound=row['lower_bound'],
                    upper_bound=row['upper_bound'],
                    source=row['source'] or '',
                    vintage=row['vintage'] or '',
                    is_available=True,
                    is_exact_match=True,
                    population_size=pop_info.get('total_size') if pop_info else None
                )
        
        return FrequencyResult(is_available=False)
    
    def estimate_joint_conservative(
        self,
        population_id: str,
        qi_values: dict[str, str]
    ) -> FrequencyResult:
        """
        Estimate joint frequency using conservative Fréchet bounds.
        
        For intersection of events A1, A2, ..., Am:
        max(0, sum(p_i) - (m-1)) <= P(intersection) <= min(p_i)
        """
        exact = self.lookup_joint(population_id, qi_values)
        if exact.is_available:
            return exact
        
        marginal_probs = []
        pop_info = self.get_population_info(population_id)
        pop_size = pop_info.get('total_size', 0) if pop_info else 0
        
        for qi_type, qi_value in qi_values.items():
            marginal = self.lookup_marginal(population_id, qi_type, qi_value)
            if not marginal.is_available:
                # Unknown value - return conservative estimate of 1
                return FrequencyResult(
                    is_available=True,
                    lower_bound=1.0,
                    upper_bound=1.0,
                    source=population_id,
                    population_size=pop_size,
                )
            marginal_probs.append(
                marginal.lower_bound
                if marginal.lower_bound is not None
                else (marginal.frequency or 0.0)
            )
        
        if not marginal_probs or pop_size == 0:
            return FrequencyResult(is_available=False)
        
        # Fréchet bounds
        m = len(marginal_probs)
        upper_bound_prob = min(marginal_probs)
        lower_bound_prob = max(0.0, sum(marginal_probs) - (m - 1))
        
        upper_bound_k = upper_bound_prob * pop_size
        lower_bound_k = lower_bound_prob * pop_size
        
        return FrequencyResult(
            count=None,
            frequency=None,
            lower_bound=lower_bound_k,
            upper_bound=upper_bound_k,
            source=population_id,
            vintage=pop_info.get('vintage', '') if pop_info else '',
            is_available=True,
            is_exact_match=False,
            population_size=pop_size
        )
    
    def add_marginal(
        self,
        population_id: str,
        qi_type: str,
        qi_value: str,
        count: int,
        frequency: float,
        source: str = "",
        vintage: str = "",
        pop_size: Optional[int] = None
    ) -> None:
        """Add a marginal frequency entry."""
        # Calculate confidence interval
        if pop_size and pop_size > 0:
            se = math.sqrt(frequency * (1 - frequency) / pop_size)
            lower_bound = max(0, frequency - 1.96 * se)
            upper_bound = min(1, frequency + 1.96 * se)
        else:
            lower_bound = frequency
            upper_bound = frequency
        
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO marginal_frequencies 
                (population_id, qi_type, qi_value, count, frequency, 
                 lower_bound, upper_bound, source, vintage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (population_id, qi_type, qi_value, count, frequency,
                  lower_bound, upper_bound, source, vintage))
    
    def add_joint(
        self,
        population_id: str,
        qi_values: dict[str, str],
        count: int,
        source: str = ""
    ) -> None:
        """Add a joint frequency entry."""
        pop_info = self.get_population_info(population_id)
        pop_size = pop_info.get('total_size', 1) if pop_info else 1
        
        frequency = count / pop_size
        se = math.sqrt(frequency * (1 - frequency) / pop_size) if pop_size > 0 else 0
        lower_bound = max(0, frequency - 1.96 * se)
        upper_bound = min(1, frequency + 1.96 * se)
        
        qi_combination = json.dumps(sorted(qi_values.keys()))
        qi_values_json = json.dumps(qi_values, sort_keys=True)
        
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO joint_frequencies
                (population_id, qi_combination, qi_values, count, frequency,
                 lower_bound, upper_bound, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (population_id, qi_combination, qi_values_json, count,
                  frequency, lower_bound, upper_bound, source))


class FrequencyTableBuilder:
    """
    Builds and refreshes frequency tables from authoritative data sources.
    
    Usage:
        builder = FrequencyTableBuilder(db_path, census_key="YOUR_KEY")
        builder.build_all()
    """
    
    def __init__(
        self,
        db_path: Path,
        census_api_key: Optional[str] = None,
        bls_api_key: Optional[str] = None,
        cdc_app_token: Optional[str] = None,
    ):
        self.table = LocalFrequencyTable(db_path)
        self.census = CensusAPIClient(census_api_key)
        self.bls = BLSAPIClient(bls_api_key)
        self.orphanet = OrphanetClient()
        self.cdc = CDCDataClient(cdc_app_token)
        
        # US adult population (18+) for rate calculations
        self.US_ADULT_POP = 258_300_000  # ~258M adults
        self.US_TOTAL_POP = 331_900_000  # ~332M total
    
    def build_all(self, population_id: str = "us_2024") -> None:
        """Build complete frequency tables from all sources."""
        logger.info("Building frequency tables from authoritative sources...")
        
        # Initialize population
        self._init_population(population_id)
        
        # Build from each source
        self._build_census_demographics(population_id)
        self._build_bls_occupations(population_id)
        self._build_orphanet_diseases(population_id)
        self._build_cdc_conditions(population_id)
        
        # Build cross-tabulations for common combinations
        self._build_joint_frequencies(population_id)
        
        logger.info("Frequency table build complete.")
    
    def _init_population(self, population_id: str) -> None:
        """Initialize population record."""
        with self.table._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO population_info
                (population_id, name, vintage, total_size, description, created_at, sources)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                population_id,
                "US Adult Population",
                "2022-2024",
                self.US_ADULT_POP,
                "US population 18+ from Census ACS, BLS, Orphanet, CDC",
                datetime.now().isoformat(),
                json.dumps(["census_acs_2022", "bls_oews_2023", "orphanet_2024", "cdc_2022"]),
            ))
    
    def _build_census_demographics(self, population_id: str) -> None:
        """Build demographic marginals from Census data."""
        logger.info("Fetching Census ACS demographics...")
        
        # Age x Sex distribution
        age_sex_data = self.census.get_age_sex_distribution()
        
        # Aggregate to marginals
        age_totals = {}
        sex_totals = {'male': 0, 'female': 0}
        
        for record in age_sex_data:
            bucket = record['age_bucket']
            sex = record['sex']
            count = record['count']
            
            age_totals[bucket] = age_totals.get(bucket, 0) + count
            sex_totals[sex] += count
        
        total = sum(sex_totals.values())
        
        # Add age marginals
        for bucket, count in age_totals.items():
            self.table.add_marginal(
                population_id, 'age_5yr', bucket, count,
                count / total if total > 0 else 0,
                source='census_acs_2022',
                vintage='2022',
                pop_size=total
            )
        
        # Add sex marginals
        for sex, count in sex_totals.items():
            self.table.add_marginal(
                population_id, 'sex', sex, count,
                count / total if total > 0 else 0,
                source='census_acs_2022',
                vintage='2022',
                pop_size=total
            )
        
        # State populations
        state_pops = self.census.get_state_populations()
        us_total = sum(state_pops.values())
        
        for state, pop in state_pops.items():
            self.table.add_marginal(
                population_id, 'state', state, pop,
                pop / us_total if us_total > 0 else 0,
                source='census_acs_2022',
                vintage='2022',
                pop_size=us_total
            )
        
        # Marital status
        marital_data = self.census.get_marital_status_distribution()
        for record in marital_data:
            self.table.add_marginal(
                population_id, 'marital_status', record['status'],
                record['count'], record['frequency'],
                source='census_acs_2022',
                vintage='2022',
                pop_size=self.US_ADULT_POP
            )
        
        # Race/ethnicity from national demographics
        national = self.census.get_national_demographics()
        if national:
            total_pop = int(national.get('B01001_001E', 0))
            
            race_mapping = {
                'white': 'B02001_002E',
                'black': 'B02001_003E',
                'asian': 'B02001_005E',
                'aian': 'B02001_004E',  # American Indian/Alaska Native
                'nhpi': 'B02001_006E',  # Native Hawaiian/Pacific Islander
                'two_or_more': 'B02001_008E',
                'hispanic': 'B03003_003E',
            }
            
            for race_key, var_code in race_mapping.items():
                count = int(national.get(var_code, 0))
                if count > 0:
                    self.table.add_marginal(
                        population_id, 'race_ethnicity', race_key, count,
                        count / total_pop if total_pop > 0 else 0,
                        source='census_acs_2022',
                        vintage='2022',
                        pop_size=total_pop
                    )
            
            # Citizenship
            citizenship_mapping = {
                'citizen': int(national.get('B05001_002E', 0)) + int(national.get('B05001_005E', 0)),
                'non_citizen': int(national.get('B05001_006E', 0)),
            }
            
            for status, count in citizenship_mapping.items():
                if count > 0:
                    self.table.add_marginal(
                        population_id, 'citizenship', status, count,
                        count / total_pop if total_pop > 0 else 0,
                        source='census_acs_2022',
                        vintage='2022',
                        pop_size=total_pop
                    )
        
        logger.info(f"Added Census demographics for {population_id}")
    
    def _build_bls_occupations(self, population_id: str) -> None:
        """Build occupation marginals from BLS data."""
        logger.info("Fetching BLS occupation data...")
        
        occupation_data = self.bls.get_occupation_employment()
        
        # Total employed (approximate from BLS)
        total_employed = 158_000_000  # ~158M employed
        
        # Map BLS categories to our normalized values
        occupation_mapping = {
            'healthcare_practitioners': 'healthcare',
            'healthcare_support': 'healthcare',
            'education': 'education',
            'zoologist': 'zoologist',
            'astronomer': 'astronomer',
            'epidemiologist': 'epidemiologist',
        }
        
        aggregated = {}
        for bls_cat, data in occupation_data.items():
            our_cat = occupation_mapping.get(bls_cat, bls_cat)
            emp = data.get('employment', 0)
            aggregated[our_cat] = aggregated.get(our_cat, 0) + emp
        
        for occupation, count in aggregated.items():
            self.table.add_marginal(
                population_id, 'occupation', occupation, count,
                count / total_employed if total_employed > 0 else 0,
                source='bls_oews_2023',
                vintage='2023',
                pop_size=total_employed
            )
        
        logger.info(f"Added BLS occupation data: {len(aggregated)} categories")
    
    def _build_orphanet_diseases(self, population_id: str) -> None:
        """Build rare disease marginals from Orphanet data."""
        logger.info("Adding Orphanet rare disease prevalence...")
        
        rare_diseases = self.orphanet.get_all_rare_diseases()
        
        for disease_key, data in rare_diseases.items():
            prevalence_per_100k = data.get('prevalence_per_100k', 0)
            frequency = prevalence_per_100k / 100_000
            count = int(frequency * self.US_ADULT_POP)
            
            self.table.add_marginal(
                population_id, 'condition', disease_key, count, frequency,
                source=f"orphanet_{data.get('orpha_code', '')}",
                vintage='2024',
                pop_size=self.US_ADULT_POP
            )
        
        logger.info(f"Added {len(rare_diseases)} rare disease entries from Orphanet")
    
    def _build_cdc_conditions(self, population_id: str) -> None:
        """Build common condition marginals from CDC data."""
        logger.info("Adding CDC condition prevalence...")
        
        conditions = self.cdc.get_all_conditions()
        
        for condition_key, data in conditions.items():
            prevalence_pct = data.get('prevalence_pct', 0)
            frequency = prevalence_pct / 100
            count = int(frequency * self.US_ADULT_POP)
            
            self.table.add_marginal(
                population_id, 'condition', condition_key, count, frequency,
                source=f"cdc_{data.get('source', 'unknown')}",
                vintage='2022',
                pop_size=self.US_ADULT_POP
            )
        
        logger.info(f"Added {len(conditions)} condition entries from CDC")
    
    def _build_joint_frequencies(self, population_id: str) -> None:
        """
        Build joint frequency entries for common QI combinations.
        
        Note: True joint frequencies require microdata (e.g., PUMS).
        Here we use Fréchet bounds for combinations we don't have
        direct cross-tabulations for.
        """
        logger.info("Building joint frequency estimates...")
        
        # For combinations we have actual cross-tabs for (from PUMS or published tables),
        # we could add them directly. For now, the system will use Fréchet bounds
        # via estimate_joint_conservative() for combinations not in the table.
        
        # Example: Add some known cross-tabulations if available
        # These would come from Census PUMS or published detailed tables
        
        # The system will automatically use conservative Fréchet bounds
        # for any combination not explicitly stored.
        
        logger.info("Joint frequency estimation configured (using Fréchet bounds)")


def create_frequency_table_from_sources(
    db_path: Path,
    census_api_key: Optional[str] = None,
    bls_api_key: Optional[str] = None,
    cdc_app_token: Optional[str] = None,
    population_id: str = "us_2024",
) -> LocalFrequencyTable:
    """
    Create and populate a frequency table from authoritative sources.
    
    Args:
        db_path: Path to SQLite database file
        census_api_key: Census Bureau API key (optional but recommended)
        bls_api_key: BLS API key (optional)
        cdc_app_token: CDC Socrata app token (optional)
        population_id: Identifier for the population dataset
    
    Returns:
        Populated LocalFrequencyTable instance
    """
    builder = FrequencyTableBuilder(
        db_path,
        census_api_key=census_api_key,
        bls_api_key=bls_api_key,
        cdc_app_token=cdc_app_token,
    )
    
    builder.build_all(population_id)
    
    return builder.table


def create_sample_frequency_table(db_path: Path) -> LocalFrequencyTable:
    """
    Create a sample frequency table with realistic data for testing.
    
    This version uses hardcoded values derived from real sources
    for offline use when API access is not available.
    """
    table = LocalFrequencyTable(db_path)
    
    US_ADULT_POP = 258_300_000
    US_TOTAL_POP = 331_900_000
    
    with table._connect() as conn:
        # Initialize population
        conn.execute("""
            INSERT OR REPLACE INTO population_info
            (population_id, name, vintage, total_size, description, created_at, sources)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'us_2024',
            "US Adult Population",
            "2022-2024",
            US_ADULT_POP,
            "US population 18+ from Census ACS, BLS, Orphanet, CDC",
            datetime.now().isoformat(),
            json.dumps(["census_acs_2022", "bls_oews_2023", "orphanet_2024", "cdc_2022"]),
        ))
    
    # === CENSUS-DERIVED DEMOGRAPHICS ===
    # Age distribution (5-year buckets, Census ACS 2022)
    age_distribution = [
        ('0-4', 0.059), ('5-9', 0.061), ('10-14', 0.064), ('15-19', 0.065),
        ('20-24', 0.066), ('25-29', 0.069), ('30-34', 0.069), ('35-39', 0.066),
        ('40-44', 0.062), ('45-49', 0.062), ('50-54', 0.063), ('55-59', 0.066),
        ('60-64', 0.064), ('65-69', 0.055), ('70-74', 0.045), ('75-79', 0.032),
        ('80-84', 0.022), ('85-89', 0.015),
    ]
    
    for bucket, freq in age_distribution:
        count = int(freq * US_TOTAL_POP)
        table.add_marginal('us_2024', 'age_5yr', bucket, count, freq,
                          'census_acs_2022', '2022', US_TOTAL_POP)
    
    # Sex distribution (Census ACS 2022)
    table.add_marginal('us_2024', 'sex', 'male', 162_000_000, 0.488,
                      'census_acs_2022', '2022', US_TOTAL_POP)
    table.add_marginal('us_2024', 'sex', 'female', 170_000_000, 0.512,
                      'census_acs_2022', '2022', US_TOTAL_POP)
    
    # Race/ethnicity (Census ACS 2022)
    race_distribution = [
        ('white', 0.758), ('black', 0.134), ('asian', 0.061),
        ('hispanic', 0.190), ('aian', 0.013), ('nhpi', 0.003),
        ('two_or_more', 0.028),
        # Specific Asian ethnicities (smaller groups for k-anonymity testing)
        ('laotian', 0.0008), ('hmong', 0.0009), ('cambodian', 0.001),
    ]
    
    for race, freq in race_distribution:
        count = int(freq * US_TOTAL_POP)
        table.add_marginal('us_2024', 'race_ethnicity', race, count, freq,
                          'census_acs_2022', '2022', US_TOTAL_POP)
    
    # State populations (Census ACS 2022, top states shown)
    state_populations = [
        ('CA', 39_040_000), ('TX', 29_530_000), ('FL', 22_240_000),
        ('NY', 19_680_000), ('PA', 12_970_000), ('IL', 12_580_000),
        ('OH', 11_760_000), ('GA', 10_910_000), ('NC', 10_700_000),
        ('MI', 10_040_000), ('NJ', 9_290_000), ('VA', 8_640_000),
        ('WA', 7_740_000), ('AZ', 7_360_000), ('MA', 7_030_000),
        ('TN', 7_050_000), ('IN', 6_810_000), ('MO', 6_180_000),
        ('MD', 6_180_000), ('WI', 5_900_000), ('CO', 5_840_000),
        ('MN', 5_710_000), ('SC', 5_280_000), ('AL', 5_070_000),
        ('LA', 4_620_000), ('KY', 4_510_000), ('OR', 4_240_000),
        ('OK', 4_000_000), ('CT', 3_630_000), ('UT', 3_380_000),
        ('PR', 3_220_000), ('IA', 3_200_000), ('NV', 3_180_000),
        ('AR', 3_040_000), ('MS', 2_940_000), ('KS', 2_940_000),
        ('NM', 2_120_000), ('NE', 1_960_000), ('ID', 1_940_000),
        ('WV', 1_780_000), ('HI', 1_440_000), ('NH', 1_400_000),
        ('ME', 1_370_000), ('MT', 1_120_000), ('RI', 1_100_000),
        ('DE', 1_000_000), ('SD', 896_000), ('ND', 780_000),
        ('AK', 733_000), ('DC', 670_000), ('VT', 647_000), ('WY', 577_000),
    ]
    
    for state, pop in state_populations:
        freq = pop / US_TOTAL_POP
        table.add_marginal('us_2024', 'state', state, pop, freq,
                          'census_acs_2022', '2022', US_TOTAL_POP)
    
    # Marital status (Census ACS 2022, 15+)
    marital_status = [
        ('single', 0.309), ('married', 0.476), ('divorced', 0.109),
        ('widowed', 0.058), ('separated', 0.018),
    ]
    
    for status, freq in marital_status:
        count = int(freq * US_ADULT_POP)
        table.add_marginal('us_2024', 'marital_status', status, count, freq,
                          'census_acs_2022', '2022', US_ADULT_POP)
    
    # Citizenship (Census ACS 2022)
    table.add_marginal('us_2024', 'citizenship', 'citizen', 308_000_000, 0.928,
                      'census_acs_2022', '2022', US_TOTAL_POP)
    table.add_marginal('us_2024', 'citizenship', 'non_citizen', 24_000_000, 0.072,
                      'census_acs_2022', '2022', US_TOTAL_POP)
    
    # === BLS OCCUPATION DATA ===
    total_employed = 158_000_000
    
    occupation_data = [
        ('healthcare', 16_500_000, 0.104),  # Healthcare practitioners + support
        ('education', 9_200_000, 0.058),
        ('management', 11_000_000, 0.070),
        ('sales', 13_500_000, 0.085),
        ('office_administrative', 18_000_000, 0.114),
        ('food_service', 12_800_000, 0.081),
        ('transportation', 11_500_000, 0.073),
        ('construction', 8_200_000, 0.052),
        ('production', 9_000_000, 0.057),
        # Rare occupations (BLS OEWS specific)
        ('zoologist', 17_500, 0.00011),
        ('astronomer', 2_500, 0.000016),
        ('epidemiologist', 8_500, 0.000054),
    ]
    
    for occupation, count, freq in occupation_data:
        table.add_marginal('us_2024', 'occupation', occupation, count, freq,
                          'bls_oews_2023', '2023', total_employed)
    
    # === CDC COMMON CONDITIONS ===
    cdc_conditions = [
        ('hypertension', 0.470),
        ('hyperlipidemia', 0.380),
        ('obesity', 0.419),
        ('diabetes_type2', 0.113),
        ('arthritis', 0.237),
        ('asthma', 0.077),
        ('depression', 0.184),
        ('anxiety', 0.156),
        ('copd', 0.045),
        ('chronic_kidney_disease', 0.140),
    ]
    
    for condition, freq in cdc_conditions:
        count = int(freq * US_ADULT_POP)
        table.add_marginal('us_2024', 'condition', condition, count, freq,
                          'cdc_nhanes_2022', '2022', US_ADULT_POP)
    
    # === ORPHANET RARE DISEASES ===
    orphanet_diseases = [
        ('ehlers_danlos', 'ORPHA:98249', 20.0),  # per 100k
        ('huntingtons', 'ORPHA:399', 7.5),
        ('marfan', 'ORPHA:558', 15.0),
        ('cystic_fibrosis', 'ORPHA:586', 28.6),
        ('pku', 'ORPHA:716', 10.0),
        ('als', 'ORPHA:803', 5.0),
        ('sickle_cell', 'ORPHA:232', 30.0),
        ('hemophilia_a', 'ORPHA:98878', 10.0),
    ]
    
    for disease, orpha_code, prev_per_100k in orphanet_diseases:
        freq = prev_per_100k / 100_000
        count = int(freq * US_ADULT_POP)
        table.add_marginal('us_2024', 'condition', disease, count, freq,
                          f'orphanet_{orpha_code}', '2024', US_ADULT_POP)
    
    # Also create hospital_2024 alias for backward compatibility
    with table._connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO population_info
            (population_id, name, vintage, total_size, description, created_at, sources)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'hospital_2024',
            "Hospital Reference Population",
            "2022-2024",
            US_ADULT_POP,
            "Alias for us_2024 for backward compatibility",
            datetime.now().isoformat(),
            json.dumps(["census_acs_2022", "bls_oews_2023", "orphanet_2024", "cdc_2022"]),
        ))
        
        # Copy all marginals to hospital_2024
        conn.execute("""
            INSERT OR REPLACE INTO marginal_frequencies
            (population_id, qi_type, qi_value, count, frequency, 
             lower_bound, upper_bound, source, vintage)
            SELECT 'hospital_2024', qi_type, qi_value, count, frequency,
                   lower_bound, upper_bound, source, vintage
            FROM marginal_frequencies
            WHERE population_id = 'us_2024'
        """)
    
    return table


# Convenience function for backward compatibility
def create_sample_frequency_table_legacy(db_path: Path) -> LocalFrequencyTable:
    """Backward-compatible alias for create_sample_frequency_table."""
    return create_sample_frequency_table(db_path)