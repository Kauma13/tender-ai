'''Orchestrates the entire pipeline as a LangGraph state machine.'''

# app/graph/tender_graph.py
from langgraph.graph import StateGraph, END, START

from app.models.state import TenderState
from app.pipeline.classify_retrieve_route import classify_retrieve_route_node
from app.pipeline.generate import generate_answers_node
from app.pipeline.verify import verify_answers_node
from app.pipeline.summarise import summarise_node  
from app.pipeline.memory import persist_memory_node 

def build_graph():
    graph = StateGraph(TenderState)

    graph.add_node("classify_retrieve_route", classify_retrieve_route_node)
    graph.add_node("generate_answers", generate_answers_node)
    graph.add_node("verify_answers", verify_answers_node)
    graph.add_node("summarise", summarise_node)
    graph.add_node("persist_memory", persist_memory_node)

    graph.add_edge(START, "classify_retrieve_route")
    graph.add_edge("classify_retrieve_route", "generate_answers")
    graph.add_edge("generate_answers", "verify_answers")
    graph.add_edge("verify_answers", "summarise")
    graph.add_edge("summarise", "persist_memory")
    graph.add_edge("persist_memory", END)

    return graph.compile()