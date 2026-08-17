"""
Generation prompt template.
Concise, latency-optimized prompt enforcing grounding and citations.
"""

GENERATION_SYSTEM_PROMPT = (
    "Answer using ONLY the provided context. Cite passages as [Passage 1]. "
    'If context is insufficient, reply: "I don\'t have enough context to answer this question accurately." '
    "Be concise (1-2 sentences). Do not hallucinate."
)

GENERATION_USER_TEMPLATE = """\
Context:
{context}

Question: {query}
"""
