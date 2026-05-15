import re
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

import config


@dataclass
class ValidationResult:
    is_grounded: bool
    overall_score: float
    sentence_scores: list[dict] = field(default_factory=list)
    citation_check: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    details: str = ""


class GroundingValidator:
    """
    Validates that LLM responses are grounded in retrieved evidence.

    Three-layer validation:
    1. Semantic similarity — embeds each response sentence and compares against source chunks
    2. Citation verification — ensures claims reference actual source documents
    3. Hallucination detection — flags sentences with no supporting evidence
    """

    def __init__(self, strictness: float = None, embed_model: SentenceTransformer = None):
        self.strictness = strictness or config.GROUNDING_STRICTNESS
        self._model = embed_model

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._model

    def validate(self, response: str, retrieved_results: list[dict]) -> ValidationResult:
        if not response or not retrieved_results:
            return ValidationResult(
                is_grounded=False,
                overall_score=0.0,
                warnings=["Empty response or no source documents provided."],
            )

        source_texts = [r["text"] for r in retrieved_results]
        available_doc_ids = {r.get("doc_id", "") for r in retrieved_results}

        sentence_scores = self._check_semantic_grounding(response, source_texts)
        citation_check = self._verify_citations(response, available_doc_ids)
        hallucination_flags = self._detect_hallucination_signals(response, source_texts)

        grounded_count = sum(
            1 for s in sentence_scores if s["score"] >= self.strictness
        )
        total_sentences = len(sentence_scores) if sentence_scores else 1
        overall_score = grounded_count / total_sentences

        warnings = []
        warnings.extend(hallucination_flags)

        if citation_check.get("invalid_citations"):
            warnings.append(
                f"Invalid citations found: {citation_check['invalid_citations']}. "
                "These reference documents not present in the retrieved context."
            )

        if overall_score < self.strictness:
            warnings.append(
                f"Overall grounding score ({overall_score:.2f}) is below "
                f"threshold ({self.strictness:.2f}). Response may contain unsupported claims."
            )

        ungrounded = [
            s for s in sentence_scores if s["score"] < self.strictness
        ]
        if ungrounded:
            warnings.append(
                f"{len(ungrounded)} of {total_sentences} sentences lack sufficient "
                f"evidence support (below {self.strictness} similarity threshold)."
            )

        is_grounded = overall_score >= self.strictness and not citation_check.get("invalid_citations")

        details = self._build_details_report(
            overall_score, sentence_scores, citation_check, warnings
        )

        return ValidationResult(
            is_grounded=is_grounded,
            overall_score=overall_score,
            sentence_scores=sentence_scores,
            citation_check=citation_check,
            warnings=warnings,
            details=details,
        )

    def _check_semantic_grounding(
        self, response: str, source_texts: list[str]
    ) -> list[dict]:
        sentences = self._split_sentences(response)
        if not sentences:
            return []

        source_embeddings = self.model.encode(source_texts, normalize_embeddings=True)
        sentence_embeddings = self.model.encode(sentences, normalize_embeddings=True)

        results = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:
                results.append({"sentence": sentence, "score": 1.0, "best_match_idx": -1})
                continue

            sim = sklearn_cosine(
                sentence_embeddings[i:i+1],
                source_embeddings,
            )
            best_idx = int(np.argmax(sim[0]))
            best_score = float(sim[0][best_idx])

            results.append({
                "sentence": sentence,
                "score": best_score,
                "best_match_idx": best_idx,
            })

        return results

    def _verify_citations(self, response: str, available_doc_ids: set) -> dict:
        citation_pattern = r"\[Source:\s*([^\]]+)\]"
        found_citations = re.findall(citation_pattern, response)
        found_citations = [c.strip() for c in found_citations]

        valid = [c for c in found_citations if c in available_doc_ids]
        invalid = [c for c in found_citations if c not in available_doc_ids]

        return {
            "total_citations": len(found_citations),
            "valid_citations": valid,
            "invalid_citations": invalid,
            "has_citations": len(found_citations) > 0,
            "all_valid": len(invalid) == 0,
        }

    def _detect_hallucination_signals(
        self, response: str, source_texts: list[str]
    ) -> list[str]:
        warnings = []
        combined_source = " ".join(source_texts).lower()

        hedging_phrases = [
            "i think", "i believe", "in my opinion", "it is generally known",
            "it is commonly accepted", "as everyone knows", "obviously",
        ]
        for phrase in hedging_phrases:
            if phrase in response.lower():
                warnings.append(
                    f"Hedging/opinion phrase detected: '{phrase}'. "
                    "Responses should be evidence-based, not opinion-based."
                )

        number_pattern = r"\b(\d+(?:\.\d+)?)\s*(%|percent|mg|mmHg|mL|kg|g/dL|mmol)"
        response_numbers = re.findall(number_pattern, response)
        for num, unit in response_numbers:
            search_term = f"{num}"
            if search_term not in combined_source:
                warnings.append(
                    f"Numeric value '{num} {unit}' in response not found in source documents. "
                    "Verify this statistic is evidence-grounded."
                )

        return warnings

    def _split_sentences(self, text: str) -> list[str]:
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if len(s.strip()) > 5]

    def _build_details_report(
        self,
        overall_score: float,
        sentence_scores: list[dict],
        citation_check: dict,
        warnings: list[str],
    ) -> str:
        lines = [
            "=== GROUNDING VALIDATION REPORT ===",
            f"Overall Grounding Score: {overall_score:.2%}",
            f"Strictness Threshold: {self.strictness:.2%}",
            f"Total Sentences Evaluated: {len(sentence_scores)}",
            "",
            "--- Citation Analysis ---",
            f"Citations Found: {citation_check.get('total_citations', 0)}",
            f"Valid Citations: {len(citation_check.get('valid_citations', []))}",
            f"Invalid Citations: {len(citation_check.get('invalid_citations', []))}",
        ]

        if warnings:
            lines.append("")
            lines.append("--- Warnings ---")
            for w in warnings:
                lines.append(f"  * {w}")

        low_score_sentences = [
            s for s in sentence_scores if s["score"] < self.strictness
        ]
        if low_score_sentences:
            lines.append("")
            lines.append("--- Weakly Grounded Sentences ---")
            for s in low_score_sentences[:5]:
                preview = s["sentence"][:80] + "..." if len(s["sentence"]) > 80 else s["sentence"]
                lines.append(f"  [{s['score']:.2f}] {preview}")

        lines.append("")
        verdict = "PASS" if overall_score >= self.strictness else "FAIL"
        lines.append(f"Verdict: {verdict}")
        lines.append("=" * 38)

        return "\n".join(lines)
