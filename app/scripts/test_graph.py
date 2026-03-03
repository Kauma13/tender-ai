# app/scripts/test_ask.py
import uuid
from app.models.state import TenderState, QuestionItem
from app.graph.tender_graph import build_graph

def main():
    graph = build_graph()

    state = TenderState(
        request_id=str(uuid.uuid4()),
        questions=[
            QuestionItem(
                question_id="Q001",
                question="Describe your organisation’s information security management approach."
            )
        ],
        retrieval_top_k=3,
        strong_threshold=0.7,
        weak_threshold=0.6,
    )

    print("\n--- Running Graph ---\n")

    final_state = graph.invoke(state)

    print(final_state)

    # for q in final_state.enriched:
    #     print("=" * 60)
    #     print("Question:", q.question)
    #     print("Domain:", q.domain)
    #     print("Route:", q.route)
    #     print("Match strength:", q.match_strength)
    #     print("Top similarity:", q.top_similarity)
    #     print("Flags:", q.flags)
    #     print("Confidence:", q.confidence)
    #     print("\nAnswer:\n", q.answer)
    #     print("=" * 60)

if __name__ == "__main__":
    main()