"""
LangGraph pipeline graph — wires all nodes together with conditional edges.

Graph shape:
  STT → Off-Topic Guard → [if relevant] → Retrieval → [if context] → Generation → Grounding Guard → END
                          [if off-topic] → END (with refusal)
                                            [if no context] → END (with fallback)
                                                                                    [if not grounded] → END (with refusal)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.generation_node import generation_node
from app.agents.guardrail_node import grounding_guard_node, off_topic_guard_node
from app.agents.retrieval_node import retrieval_node
from app.agents.state import PipelineState
from app.agents.stt_node import stt_node


def _route_off_topic(state: PipelineState) -> str:
    """Conditional edge: route off-topic queries to END."""
    if state.get("is_off_topic", False):
        return "end_off_topic"
    return "retrieval"


def _route_context(state: PipelineState) -> str:
    """Conditional edge: route to generation only if we have enough context."""
    if not state.get("has_sufficient_context", False):
        return "end_no_context"
    return "generation"


def _route_grounding(state: PipelineState) -> str:
    """Conditional edge: block ungrounded answers."""
    if not state.get("is_grounded", True):
        return "end_not_grounded"
    return "end_success"


def build_rag_graph() -> StateGraph:
    """
    Construct and compile the full RAG pipeline as a LangGraph StateGraph.

    Returns a compiled graph ready for invocation.
    """
    graph = StateGraph(PipelineState)

    # ── Add nodes ───────────────────────────────────────────
    graph.add_node("stt", stt_node)
    graph.add_node("off_topic_guard", off_topic_guard_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generation", generation_node)
    graph.add_node("grounding_guard", grounding_guard_node)

    # ── Wire edges ──────────────────────────────────────────
    graph.set_entry_point("stt")

    graph.add_edge("stt", "off_topic_guard")

    graph.add_conditional_edges(
        "off_topic_guard",
        _route_off_topic,
        {
            "retrieval": "retrieval",
            "end_off_topic": END,
        },
    )

    graph.add_conditional_edges(
        "retrieval",
        _route_context,
        {
            "generation": "generation",
            "end_no_context": END,
        },
    )

    graph.add_edge("generation", "grounding_guard")

    graph.add_conditional_edges(
        "grounding_guard",
        _route_grounding,
        {
            "end_success": END,
            "end_not_grounded": END,
        },
    )

    return graph.compile()
