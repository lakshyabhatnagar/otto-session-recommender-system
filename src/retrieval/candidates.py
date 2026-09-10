import polars as pl


def generate_candidates(session_events: pl.DataFrame, top_covisitation: pl.DataFrame) -> pl.DataFrame:
    neighbors = {}

    for row in top_covisitation.iter_rows(named=True):
        aid = row["aid_x"]
        if aid not in neighbors:
            neighbors[aid] = []

        neighbors[aid].append((row["aid_y"], row["score"]))

    sessions = (
        session_events
        .sort(["session", "ts"])
        .group_by("session")
        .agg(pl.col("aid").unique(maintain_order=True).alias("items"))
    )

    output = []

    for session, items in sessions.iter_rows():
        candidates = set(items)
        support = {}

        for aid in items:
            for neighbor, score in neighbors.get(aid, []):
                candidates.add(neighbor)
                support[neighbor] = support.get(neighbor, 0) + 1

        for candidate in candidates:
            output.append({
                "session": session,
                "candidate": candidate,
                "support": support.get(candidate, 0)
            })

    return pl.DataFrame(output)