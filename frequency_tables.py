"""
frequency_tables.py - Local versioned population frequency data

Replaces live API calls with offline-refreshed local tables.
Provides joint frequency estimation with conservative bounds.
"""

import sqlite3
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager
import math


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


class LocalFrequencyTable:
    """
    Local SQLite-based frequency tables for joint QI combinations.
    
    Tables are built offline from:
    - Census ACS PUMS (age, sex, geography, occupation)
    - Hospital patient demographics (if available)
    - Disease prevalence registries
    
    This avoids:
    1. Unpredictable latency from live API calls
    2. Leaking query values to external services
    3. The impossibility of getting joint distributions from separate APIs
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS population_info (
        population_id TEXT PRIMARY KEY,
        name TEXT,
        vintage TEXT,
        total_size INTEGER,
        description TEXT,
        created_at TEXT
    );
    
    CREATE TABLE IF NOT EXISTS joint_frequencies (
        population_id TEXT,
        qi_combination TEXT,  -- JSON array of QI types
        qi_values TEXT,       -- JSON object of normalized values
        count INTEGER,
        frequency REAL,
        lower_bound REAL,
        upper_bound REAL,
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
        FOREIGN KEY (population_id) REFERENCES population_info(population_id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_marginal_lookup
    ON marginal_frequencies(population_id, qi_type, qi_value);
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
        """Get metadata about a reference population."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM population_info WHERE population_id = ?",
                (population_id,)
            ).fetchone()
            return dict(row) if row else None
    
    def lookup_joint(
        self, 
        population_id: str,
        qi_values: dict[str, str]  # {qi_type: normalized_value}
    ) -> FrequencyResult:
        """
        Look up the joint frequency for a complete QI combination.
        
        This is the preferred method - use actual joint counts, not
        multiplied marginals.
        """
        qi_combination = json.dumps(sorted(qi_values.keys()))
        qi_values_json = json.dumps(qi_values, sort_keys=True)
        
        with self._connect() as conn:
            # First try exact match
            row = conn.execute("""
                SELECT count, frequency, lower_bound, upper_bound
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
                    source=population_id,
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
                SELECT count, frequency, lower_bound, upper_bound
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
                    source=population_id,
                    vintage=pop_info.get('vintage', '') if pop_info else '',
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
        
        This is used when we don't have actual joint counts.
        The lower bound is conservative - if it can't establish k >= k_min,
        we haven't demonstrated safety.
        """
        # First try exact joint lookup
        exact = self.lookup_joint(population_id, qi_values)
        if exact.is_available:
            return exact
        
        # Fall back to conservative bounds from marginals
        marginal_probs = []
        pop_info = self.get_population_info(population_id)
        pop_size = pop_info.get('total_size', 0) if pop_info else 0
        
        for qi_type, qi_value in qi_values.items():
            marginal = self.lookup_marginal(population_id, qi_type, qi_value)
            if not marginal.is_available:
                # Fallback for unseen values: use a very conservative count of 1
                # so the router still gets a numeric k_lower instead of failing closed.
                return FrequencyResult(
                    is_available=True,
                    lower_bound=1.0,
                    upper_bound=1.0,
                    source=population_id,
                    population_size=pop_size,
                )
            # A defensible lower bound for the intersection must use the
            # marginal lower confidence bounds. Using point estimates here
            # overstates the amount of anonymity established by the table.
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
        
        # Convert to counts
        upper_bound_k = upper_bound_prob * pop_size
        lower_bound_k = lower_bound_prob * pop_size
        
        return FrequencyResult(
            count=None,  # Not a direct count
            frequency=None,
            lower_bound=lower_bound_k,
            upper_bound=upper_bound_k,
            source=population_id,
            vintage=pop_info.get('vintage', '') if pop_info else '',
            is_available=True,
            is_exact_match=False,
            population_size=pop_size
        )


def create_sample_frequency_table(db_path: Path):
    """
    Create a sample frequency table for testing.
    
    In production, this would be populated from:
    - Census ACS PUMS microdata
    - Hospital EHR demographics (aggregated, not individual)
    - Disease registries
    - BLS occupation data
    
    Refreshed offline, versioned, and validated before deployment.
    """
    table = LocalFrequencyTable(db_path)
    
    with table._connect() as conn:
        def add_marginal(pop_id: str, qi_type: str, qi_value: str, count: int, freq: float):
            se = math.sqrt(freq * (1 - freq) / 150000)
            lower = max(0, freq - 1.96 * se)
            upper = min(1, freq + 1.96 * se)
            conn.execute(
                """
                INSERT OR REPLACE INTO marginal_frequencies 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (pop_id, qi_type, qi_value, count, freq, lower, upper),
            )

        def add_joint(pop_id: str, qi_vals: dict[str, str], count: int):
            pop_size = 150000
            freq = count / pop_size
            se = math.sqrt(freq * (1 - freq) / pop_size)
            conn.execute(
                """
                INSERT OR REPLACE INTO joint_frequencies VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pop_id,
                    json.dumps(sorted(qi_vals.keys())),
                    json.dumps(qi_vals, sort_keys=True),
                    count,
                    freq,
                    max(0, freq - 1.96 * se),
                    min(1, freq + 1.96 * se),
                ),
            )

        # Sample population
        conn.execute("""
            INSERT OR REPLACE INTO population_info VALUES
            ('hospital_2024', 'Regional Hospital 2024', '2024-Q1', 150000, 
             'Patient population for regional hospital system', datetime('now'))
        """)
        
        # Sample marginal frequencies covering the benchmark's census-style QIs.
        age_buckets = [
            ("0-4", 0.030), ("5-9", 0.035), ("10-14", 0.040), ("15-19", 0.045),
            ("20-24", 0.050), ("25-29", 0.055), ("30-34", 0.060), ("35-39", 0.065),
            ("40-44", 0.070), ("45-49", 0.075), ("50-54", 0.080), ("55-59", 0.075),
            ("60-64", 0.065), ("65-69", 0.055), ("70-74", 0.045), ("75-79", 0.035),
            ("80-84", 0.025), ("85-89", 0.020), ("90-94", 0.015), ("95-99", 0.010),
        ]
        sexes = [
            ("male", 0.48),
            ("female", 0.52),
        ]
        marital_statuses = [
            ("married", 0.40),
            ("single", 0.33),
            ("divorced", 0.12),
            ("widowed", 0.08),
            ("separated", 0.07),
        ]
        citizenships = [
            ("citizen", 0.85),
            ("naturalized", 0.08),
            ("noncitizen", 0.05),
            ("us_born", 0.78),
        ]
        race_ethnicities = [
            ("white", 0.60),
            ("black", 0.13),
            ("asian", 0.06),
            ("hispanic", 0.19),
            ("native_american", 0.02),
            ("pacific_islander", 0.01),
            ("laotian", 0.002),
            ("hmong", 0.002),
        ]
        states = [
            ("AL", 0.017), ("AK", 0.012), ("AZ", 0.020), ("AR", 0.016), ("CA", 0.023),
            ("CO", 0.019), ("CT", 0.015), ("DE", 0.010), ("DC", 0.008), ("FL", 0.022),
            ("GA", 0.021), ("HI", 0.012), ("ID", 0.014), ("IL", 0.020), ("IN", 0.018),
            ("IA", 0.015), ("KS", 0.015), ("KY", 0.017), ("LA", 0.017), ("ME", 0.011),
            ("MD", 0.019), ("MA", 0.017), ("MI", 0.019), ("MN", 0.018), ("MS", 0.016),
            ("MO", 0.018), ("MT", 0.011), ("NE", 0.013), ("NV", 0.014), ("NH", 0.011),
            ("NJ", 0.019), ("NM", 0.014), ("NY", 0.023), ("NC", 0.021), ("ND", 0.010),
            ("OH", 0.020), ("OK", 0.016), ("OR", 0.015), ("PA", 0.020), ("RI", 0.010),
            ("SC", 0.017), ("SD", 0.010), ("TN", 0.018), ("TX", 0.024), ("UT", 0.015),
            ("VT", 0.009), ("VA", 0.019), ("WA", 0.018), ("WV", 0.010), ("WI", 0.017),
            ("WY", 0.008),
        ]

        for bucket, freq in age_buckets:
            add_marginal('hospital_2024', 'age_5yr', bucket, int(freq * 150000), freq)

        for value, freq in sexes:
            add_marginal('hospital_2024', 'sex', value, int(freq * 150000), freq)

        for value, freq in marital_statuses:
            add_marginal('hospital_2024', 'marital_status', value, int(freq * 150000), freq)

        for value, freq in citizenships:
            add_marginal('hospital_2024', 'citizenship', value, int(freq * 150000), freq)

        for value, freq in race_ethnicities:
            add_marginal('hospital_2024', 'race_ethnicity', value, max(1, int(freq * 150000)), freq)

        for value, freq in states:
            add_marginal('hospital_2024', 'state', value, int(freq * 150000), freq)

        # Domain-specific conditions and occupations still used by the high-harm checks.
        for pop_id, qi_type, qi_value, count, freq in [
            ('hospital_2024', 'condition', 'diabetes_type2', 22500, 0.15),
            ('hospital_2024', 'condition', 'hypertension', 45000, 0.30),
            ('hospital_2024', 'condition', 'hyperlipidemia', 37500, 0.25),
            ('hospital_2024', 'condition', 'ehlers_danlos', 150, 0.001),
            ('hospital_2024', 'condition', 'huntingtons', 45, 0.0003),
            ('hospital_2024', 'occupation', 'healthcare', 15000, 0.10),
            ('hospital_2024', 'occupation', 'education', 12000, 0.08),
            ('hospital_2024', 'occupation', 'zoologist', 15, 0.0001),
        ]:
            add_marginal(pop_id, qi_type, qi_value, count, freq)
        
        # Sample joint frequencies for common combinations
        # In practice, these come from cross-tabulations of the reference data
        for age_bucket, age_freq in age_buckets:
            for sex_value, sex_freq in sexes:
                for state_value, state_freq in states:
                    count = max(1, int(150000 * age_freq * sex_freq * state_freq))
                    add_joint('hospital_2024', {
                        'age_5yr': age_bucket,
                        'sex': sex_value,
                        'state': state_value,
                    }, count)

        for age_bucket, age_freq in age_buckets:
            for sex_value, sex_freq in sexes:
                for marital_value, marital_freq in marital_statuses:
                    count = max(1, int(150000 * age_freq * sex_freq * marital_freq))
                    add_joint('hospital_2024', {
                        'age_5yr': age_bucket,
                        'sex': sex_value,
                        'marital_status': marital_value,
                    }, count)

        for age_bucket, age_freq in age_buckets:
            for sex_value, sex_freq in sexes:
                for citizenship_value, citizenship_freq in citizenships:
                    count = max(1, int(150000 * age_freq * sex_freq * citizenship_freq))
                    add_joint('hospital_2024', {
                        'age_5yr': age_bucket,
                        'sex': sex_value,
                        'citizenship': citizenship_value,
                    }, count)

        for age_bucket, age_freq in age_buckets:
            for sex_value, sex_freq in sexes:
                for race_value, race_freq in race_ethnicities:
                    count = max(1, int(150000 * age_freq * sex_freq * race_freq))
                    add_joint('hospital_2024', {
                        'age_5yr': age_bucket,
                        'sex': sex_value,
                        'race_ethnicity': race_value,
                    }, count)

        # A couple of explicit cross-tabs for the earlier sample coverage.
        add_joint('hospital_2024', {
            'age_5yr': '45-49', 'sex': 'male', 'state': 'MD'
        }, 4400)
        add_joint('hospital_2024', {
            'age_5yr': '45-49', 'sex': 'male', 'condition': 'diabetes_type2'
        }, 830)
    
    return table
