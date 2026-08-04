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
        except Exception as e:
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
            if "UNUSUAL_EVENT" in factors:
                f = factors["UNUSUAL_EVENT"]
                evidence.unusual_event = f.get("detected", False)
                evidence.unusual_event_confidence = f.get("confidence", 0.0)
            
            if "PUBLIC_SEARCHABLE_EVENT" in factors:
                f = factors["PUBLIC_SEARCHABLE_EVENT"]
                evidence.public_searchable_event = f.get("detected", False)
                evidence.public_event_confidence = f.get("confidence", 0.0)
            
            if "SMALL_COMMUNITY" in factors:
                f = factors["SMALL_COMMUNITY"]
                evidence.small_community = f.get("detected", False)
                evidence.small_community_confidence = f.get("confidence", 0.0)
            
            if "TEMPORAL_CORRELATION" in factors:
                f = factors["TEMPORAL_CORRELATION"]
                evidence.temporal_correlation_risk = f.get("detected", False)
                evidence.temporal_confidence = f.get("confidence", 0.0)
            
            if "RELATIONSHIP_NETWORK" in factors:
                f = factors["RELATIONSHIP_NETWORK"]
                evidence.relationship_network_risk = f.get("detected", False)
                evidence.relationship_confidence = f.get("confidence", 0.0)
            
            if "INFERENTIAL_MEDICAL" in factors:
                f = factors["INFERENTIAL_MEDICAL"]
                evidence.inferential_medical_disclosure = f.get("detected", False)
                evidence.inferential_confidence = f.get("confidence", 0.0)
            
            evidence.overall_confidence = data.get("overall_confidence", 0.0)
            
            # Check for abstention (all low confidence)
            all_confidences = [
                evidence.unusual_event_confidence,
                evidence.public_event_confidence,
                evidence.small_community_confidence,
                evidence.temporal_confidence,
                evidence.relationship_confidence,
                evidence.inferential_confidence,
            ]
            if all(c < 0.3 for c in all_confidences) and evidence.overall_confidence < 0.3:
                evidence.model_abstained = True
            
        except json.JSONDecodeError:
            evidence.parsing_error = True
        except Exception:
            evidence.parsing_error = True
        
        return evidence