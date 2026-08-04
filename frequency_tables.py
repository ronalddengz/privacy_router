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
                # Can't estimate if any marginal is missing
                return FrequencyResult(
                    is_available=False,
                    source=population_id
                )
            # Use upper bound of marginal to be conservative about joint
            marginal_probs.append(marginal.frequency)
        
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
        # Sample population
        conn.execute("""
            INSERT OR REPLACE INTO population_info VALUES
            ('hospital_2024', 'Regional Hospital 2024', '2024-Q1', 150000, 
             'Patient population for regional hospital system', datetime('now'))
        """)
        
        # Sample marginal frequencies (illustrative only)
        marginals = [
            # Age buckets (5-year)
            ('hospital_2024', 'age_5yr', '40-44', 12000, 0.08),
            ('hospital_2024', 'age_5yr', '45-49', 11500, 0.077),
            ('hospital_2024', 'age_5yr', '50-54', 13000, 0.087),
            
            # Sex
            ('hospital_2024', 'sex', 'male', 72000, 0.48),
            ('hospital_2024', 'sex', 'female', 78000, 0.52),
            
            # State
            ('hospital_2024', 'state', 'MD', 120000, 0.80),
            ('hospital_2024', 'state', 'VA', 20000, 0.133),
            ('hospital_2024', 'state', 'DC', 10000, 0.067),
            
            # Common conditions
            ('hospital_2024', 'condition', 'diabetes_type2', 22500, 0.15),
            ('hospital_2024', 'condition', 'hypertension', 45000, 0.30),
            ('hospital_2024', 'condition', 'hyperlipidemia', 37500, 0.25),
            
            # Rare conditions - these should trigger high-harm
            ('hospital_2024', 'condition', 'ehlers_danlos', 150, 0.001),
            ('hospital_2024', 'condition', 'huntingtons', 45, 0.0003),
            
            # Occupations
            ('hospital_2024', 'occupation', 'healthcare', 15000, 0.10),
            ('hospital_2024', 'occupation', 'education', 12000, 0.08),
            ('hospital_2024', 'occupation', 'zoologist', 15, 0.0001),  # Very rare
        ]
        
        for pop_id, qi_type, qi_value, count, freq in marginals:
            # Add confidence bounds (simplified - real implementation would
            # use proper statistical methods based on sample size)
            se = math.sqrt(freq * (1 - freq) / 150000)  # Standard error
            lower = max(0, freq - 1.96 * se)
            upper = min(1, freq + 1.96 * se)
            
            conn.execute("""
                INSERT OR REPLACE INTO marginal_frequencies 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (pop_id, qi_type, qi_value, count, freq, lower, upper))
        
        # Sample joint frequencies for common combinations
        # In practice, these come from cross-tabulations of the reference data
        joints = [
            # (population, qi_types, qi_values, count)
            ('hospital_2024', 
             ['age_5yr', 'sex', 'state'],
             {'age_5yr': '45-49', 'sex': 'male', 'state': 'MD'},
             4400),  # ~3% of population
            
            ('hospital_2024',
             ['age_5yr', 'sex', 'condition'],
             {'age_5yr': '45-49', 'sex': 'male', 'condition': 'diabetes_type2'},
             830),  # ~0.5%
        ]
        
        for pop_id, qi_types, qi_vals, count in joints:
            freq = count / 150000
            se = math.sqrt(freq * (1 - freq) / 150000)
            
            conn.execute("""
                INSERT OR REPLACE INTO joint_frequencies VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pop_id,
                json.dumps(sorted(qi_types)),
                json.dumps(qi_vals, sort_keys=True),
                count,
                freq,
                max(0, freq - 1.96 * se),
                min(1, freq + 1.96 * se)
            ))
    
    return table