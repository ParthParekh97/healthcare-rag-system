import config


class ContextBuilder:
    """
    Implements context engineering for the RAG pipeline.

    Constructs a structured prompt that:
    1. Defines the system role and behavioral constraints
    2. Injects retrieved evidence with source attribution
    3. Adds grounding instructions to prevent hallucination
    4. Formats the user query with answering guidelines
    """

    SYSTEM_PROMPT = (
        "You are a clinical knowledge assistant specializing in healthcare. "
        "You answer questions ONLY using the provided reference documents. "
        "You must follow these rules strictly:\n"
        "1. Base every claim on the provided evidence passages.\n"
        "2. Cite the source document for each factual statement using [Source: <doc_id>] format.\n"
        "3. If the provided documents do not contain enough information to answer, "
        "explicitly state: 'The provided documents do not contain sufficient information to answer this question.'\n"
        "4. Never fabricate medical data, statistics, dosages, or clinical guidelines.\n"
        "5. When summarizing, stay faithful to the source material — do not add external knowledge.\n"
        "6. Structure your response clearly with sections if the answer is multi-part."
    )

    def build_prompt(self, query: str, retrieved_results: list[dict]) -> list[dict]:
        evidence_block = self._format_evidence(retrieved_results)
        grounding_instruction = self._build_grounding_instruction(retrieved_results)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{grounding_instruction}\n\n"
                    f"--- REFERENCE DOCUMENTS ---\n\n"
                    f"{evidence_block}\n\n"
                    f"--- END OF REFERENCE DOCUMENTS ---\n\n"
                    f"QUESTION: {query}\n\n"
                    f"Provide a thorough, evidence-grounded answer. "
                    f"Cite sources for every factual claim using [Source: <doc_id>]. "
                    f"If the documents lack sufficient information, say so explicitly."
                ),
            },
        ]
        return messages

    def build_summarization_prompt(self, query: str, retrieved_results: list[dict]) -> list[dict]:
        evidence_block = self._format_evidence(retrieved_results)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"--- REFERENCE DOCUMENTS ---\n\n"
                    f"{evidence_block}\n\n"
                    f"--- END OF REFERENCE DOCUMENTS ---\n\n"
                    f"SUMMARIZATION REQUEST: {query}\n\n"
                    f"Provide a concise, faithful summary of the relevant information "
                    f"from the documents above. Cite sources using [Source: <doc_id>]. "
                    f"Do not introduce information beyond what is present in the documents."
                ),
            },
        ]
        return messages

    def _format_evidence(self, results: list[dict]) -> str:
        if not results:
            return "[No relevant documents found]"

        blocks = []
        for i, result in enumerate(results, 1):
            doc_id = result.get("doc_id", "unknown")
            title = result.get("metadata", {}).get("title", "Untitled")
            section = result.get("metadata", {}).get("section", "")
            score = result.get("score", 0.0)
            text = result.get("text", "")

            header = f"[Evidence {i}] Document: {doc_id} | Title: {title}"
            if section:
                header += f" | Section: {section}"
            header += f" | Relevance Score: {score:.3f}"

            blocks.append(f"{header}\n{text}")

        return "\n\n".join(blocks)

    def _build_grounding_instruction(self, results: list[dict]) -> str:
        doc_ids = list({r.get("doc_id", "") for r in results if r.get("doc_id")})
        avg_score = (
            sum(r.get("score", 0) for r in results) / len(results)
            if results
            else 0
        )

        instruction = "GROUNDING CONTEXT:\n"
        instruction += f"- {len(results)} evidence passages retrieved from {len(doc_ids)} document(s).\n"
        instruction += f"- Average relevance score: {avg_score:.3f}\n"
        instruction += f"- Available source documents: {', '.join(doc_ids)}\n"

        if avg_score < config.SIMILARITY_THRESHOLD:
            instruction += (
                "- WARNING: Retrieval confidence is low. The documents may not "
                "contain directly relevant information. Acknowledge uncertainty in your response."
            )

        return instruction
