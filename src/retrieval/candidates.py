import polars as pl


def matrix_to_lookup(matrix: pl.DataFrame) -> dict[int, list[tuple[int, float]]]:
    lookup = {}

    for aid_x, aid_y, score in matrix.select("aid_x", "aid_y", "score").iter_rows():
        if aid_x not in lookup:
            lookup[aid_x] = []

        lookup[aid_x].append((aid_y, score))

    return lookup


def generate_candidates(
    session_events: pl.DataFrame,
    general_matrix: pl.DataFrame,
    type_matrix: pl.DataFrame,
    time_matrix: pl.DataFrame,
    buy_matrix: pl.DataFrame,
    general_k: int = 50,
    type_k: int = 20,
    time_k: int = 20,
    buy_k: int = 20
) -> pl.DataFrame:

    general_lookup = matrix_to_lookup(general_matrix)
    type_lookup = matrix_to_lookup(type_matrix)
    time_lookup = matrix_to_lookup(time_matrix)
    buy_lookup = matrix_to_lookup(buy_matrix)

    sessions = (
        session_events
        .sort(["session", "ts"], descending=[False, True])
        .group_by("session", maintain_order=True)
        .agg(pl.col("aid").unique(maintain_order=True).alias("items"))
    )

    output = []

    for session, items in sessions.iter_rows():

        candidate_data = {}

        # Always include products already present in the session
        for aid in items:
            candidate_data[aid] = {
                "general_score": 0.0,
                "type_score": 0.0,
                "time_score": 0.0,
                "buy_score": 0.0
            }

        matrices = [
            ("general_score", general_lookup, general_k),
            ("type_score", type_lookup, type_k),
            ("time_score", time_lookup, time_k),
            ("buy_score", buy_lookup, buy_k)
        ]

        for score_name, lookup, top_k in matrices:
            scores = {}

            # Aggregate evidence from all actual session items
            for aid in items:
                for candidate, score in lookup.get(aid, []):
                    scores[candidate] = scores.get(candidate, 0) + score

            # Keep only the best candidates from this source
            best_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            for candidate, score in best_candidates:
                if candidate not in candidate_data:
                    candidate_data[candidate] = {
                        "general_score": 0.0,
                        "type_score": 0.0,
                        "time_score": 0.0,
                        "buy_score": 0.0
                    }

                candidate_data[candidate][score_name] = score

        for candidate, scores in candidate_data.items():
            output.append({
                "session": session,
                "candidate": candidate,
                **scores
            })

    return pl.DataFrame(output)
