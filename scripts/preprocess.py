import pandas as pd
import polars as pl
import json
import random




def sample_jsonl_sessions(
    input_path: str,
    output_path: str,
    n_sessions: int = 100_000,
    seed: int = 42,
):
    random.seed(seed)

    reservoir = []

    with open(input_path, "r") as f:
        for i, line in enumerate(f):

            session = json.loads(line)

            if i < n_sessions:
                reservoir.append(session)

            else:
                j = random.randint(0, i)

                if j < n_sessions:
                    reservoir[j] = session

    rows = []

    for session in reservoir:
        session_id = session["session"]

        for event in session["events"]:
            rows.append(
                {
                    "session": session_id,
                    "aid": event["aid"],
                    "ts": event["ts"],
                    "type": event["type"],
                }
            )

    df = pl.DataFrame(rows)

    df.write_parquet(output_path)

sample_jsonl_sessions(
input_path="data/raw/train.jsonl",
output_path="data/processed/train_dev.parquet",
n_sessions=100_000,
seed=42,
)