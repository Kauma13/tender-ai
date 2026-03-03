# app/scripts/ingest_historical.py
import pandas as pd
import argparse
from typing import List, Dict
from dataclasses import dataclass

from app.core.llm import get_embeddings
from app.rag.chroma_store import init_stores, ingest_historical_rows, collection_count
from app.core.logger import logger

@dataclass(frozen=True)
class HistoricalRecord:
    source_id: str
    domain: str
    question: str
    answer: str

def load_historical_records(df: pd.DataFrame) -> List[HistoricalRecord]:
    records: List[HistoricalRecord] = []
    for _, row in df.iterrows():
        q = str(row["Question"]).strip()
        a = str(row["Answer"]).strip()
        if not q or not a:
            continue
        records.append(
            HistoricalRecord(
                source_id=str(row["Source_ID"]).strip(),
                domain=str(row["Domain"]).strip(),
                question=q,
                answer=a,
            )
        )
    return records

def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df

def main():
    parser = argparse.ArgumentParser(description="Ingest historical tender Q/A into Chroma")
    parser.add_argument("--csv", required=True, help="Path to CSV with columns: source_id,domain,question,answer")
    args = parser.parse_args()

    logger.info("OpenAI embeddings initialized")
    embeddings = get_embeddings()
    hist_vs, _gen_vs = init_stores(embeddings)

    hist_df = load_csv(args.csv)
    rows = load_historical_records(hist_df)
    added = ingest_historical_rows(rows, hist_vs)

    print(f"Loaded rows: {len(rows)}")
    print(f"Added docs:  {added}")
    print(f"Collection count now: {collection_count(hist_vs)}")

if __name__ == "__main__":
    main()