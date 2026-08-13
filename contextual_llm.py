"""
contextual_llm.py - Local LLM for contextual analysis

This is invoked ONLY when the fast gate is uncertain.
It analyzes already-masked text for contextual risks that
deterministic detectors miss.
"""

import json
import requests
from dataclasses import dataclass
from typing import Optional

from policy import ContextualEvidence


class ContextualLLMAnalyzer:
    """
    Local LLM analyzer for contextual sensitivity.
    
    Only invoked when:
    - Gate is uncertain
    - Text has been masked
    - Need to assess residual contextual risk
    
    This is the exception handler, not the default path.
    """
    
    SENSITIVITY_FACTORS = [
        {
            "name": "UNUSUAL_EVENT",
            "description": "Narrative describes unusual or unique events that could identify someone"
        },
        {
            "name": "PUBLIC_SEARCHABLE_EVENT",
            "description": "Events that could be found via news search or public records"
        },
        {
            "name": "SMALL_COMMUNITY",
            "description": "Context suggests a small community where individuals are easily identified"
        },
        {
            "name": "TEMPORAL_CORRELATION",
            "description": "Timestamps or sequences could be matched to external records"
        },
        {
            "name": "RELATIONSHIP_NETWORK",
            "description": "Relationships described could help identify the person"
        },
        {
            "name": "INFERENTIAL_MEDICAL",
            "description": "Sensitive medical facts can be inferred from symptoms/treatments described"
        },
    ]
    
    def __init__(
        self,
        model: str = "phi3.5:3.8b",
        ollama_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: int = 120
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.timeout = timeout
    
    def analyze(self, masked_text: str) -> ContextualEvidence:
        """
        Analyze MASKED text for contextual sensitivity.
        
        Important: This operates on already-masked text.
        Bracketed placeholders like [PERSON] should be ignored.
        """
        evidence = ContextualEvidence()
        evidence.analysis_performed = True
        
        prompt = self._build_prompt(masked_text)
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "format": "json",
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("response", "")
            
            return self._parse_response(response_text)
            
        except requests.exceptions.Timeout:
            evidence.model_abstained = True
            return evidence
        except Exception:
            evidence.model_abstained = True
            evidence.parsing_error = True
            return evidence
    
    def _build_prompt(self, text: str) -> str:
        factors_desc = "\n".join([
            f"- {f['name']}: {f['description']}"
            for f in self.SENSITIVITY_FACTORS
        ])
        
        return f"""You are a privacy analyst. The text below has ALREADY been redacted.
Text inside square brackets like [PERSON] or [LOCATION] are PLACEHOLDERS for removed data.
You MUST IGNORE all bracketed text completely - only analyze the unredacted words.

Assess these contextual risk factors using ONLY the unredacted content:
{factors_desc}

For each factor, provide:
- detected: true/false
- confidence: 0.0 to 1.0
- explanation: brief reason

If evidence comes only from a bracketed placeholder, set detected=false.

Return valid JSON with this structure:
{{
  "factors": {{
    "UNUSUAL_EVENT": {{"detected": false, "confidence": 0.0, "explanation": "..."}},
    ...
  }},
  "overall_confidence": 0.0
}}

Text to analyze:
\"\"\"{text}\"\"\"
"""
    
    def _parse_response(self, response_text: str) -> ContextualEvidence:
        """Parse LLM response into structured evidence."""
        evidence = ContextualEvidence()
        evidence.analysis_performed = True

        def read_factor(factors: dict, name: str) -> tuple[bool, float]:
            value = factors.get(name, {})
            if not isinstance(value, dict):
                return False, 0.0
            detected = value.get("detected", False)
            confidence = value.get("confidence", 0.0)
            if not isinstance(detected, bool):
                detected = False
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            return detected, confidence
        
        try:
            # Try to extract JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                data = json.loads(json_str)
            else:
                evidence.parsing_error = True
                return evidence
            
            factors = data.get("factors", {})
            
            # Map to evidence fields
            (evidence.unusual_event,
             evidence.unusual_event_confidence) = read_factor(factors, "UNUSUAL_EVENT")
            (evidence.public_searchable_event,
             evidence.public_event_confidence) = read_factor(factors, "PUBLIC_SEARCHABLE_EVENT")
            (evidence.small_community,
             evidence.small_community_confidence) = read_factor(factors, "SMALL_COMMUNITY")
            (evidence.temporal_correlation_risk,
             evidence.temporal_confidence) = read_factor(factors, "TEMPORAL_CORRELATION")
            (evidence.relationship_network_risk,
             evidence.relationship_confidence) = read_factor(factors, "RELATIONSHIP_NETWORK")
            (evidence.inferential_medical_disclosure,
             evidence.inferential_confidence) = read_factor(factors, "INFERENTIAL_MEDICAL")

            try:
                evidence.overall_confidence = float(data.get("overall_confidence", 0.0))
            except (TypeError, ValueError):
                evidence.overall_confidence = 0.0
            evidence.overall_confidence = max(0.0, min(1.0, evidence.overall_confidence))
            
            # Check for abstention (all low confidence)
            all_confidences = [
                evidence.unusual_event_confidence,
                evidence.public_event_confidence,
                evidence.small_community_confidence,
                evidence.temporal_confidence,
                evidence.relationship_confidence,
                evidence.inferential_confidence,
            ]
            #if all(c < 0.3 for c in all_confidences) and evidence.overall_confidence < 0.3:
            #    evidence.model_abstained = True
            
        except json.JSONDecodeError:
            evidence.parsing_error = True
        except Exception:
            evidence.parsing_error = True
        
        return evidence
