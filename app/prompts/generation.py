"""
Generation prompt template.
Concise, latency-optimized prompt enforcing grounding and citations.
"""

GENERATION_SYSTEM_PROMPT = (
    "Answer strictly using ONLY provided context in 1-2 concise sentences. "
    "Cite sources as [Passage N]. If context is insufficient, reply: 'Insufficient context to answer.'"
)

GENERATION_USER_TEMPLATE = """\
Context:
{context}

Question: {query}
"""
