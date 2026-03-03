# app/scripts/retrieve_test.py
import argparse
from app.rag.retriever import retrieve_matches

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q", required=True, help="query text")
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    matches = retrieve_matches(args.q, args.k)
    for i, m in enumerate(matches, 1):
        print(f"\n[{i}] sim={m.similarity:.3f} domain={m.domain} source={m.source_id}")
        print("Q:", m.historical_question)
        print("A:", m.historical_answer[:300], "..." if len(m.historical_answer) > 300 else "")

if __name__ == "__main__":
    main()