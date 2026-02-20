import pandas as pd
import ast
import json

IN_PATH = "data/raw/mturk_aggregate.csv"
OUT_PATH = "data/processed/mturk_seeds.csv"

N_PER_CONDITION = 20  # 최종 seed 개수(쌍 기준이면 20 ids = 40 rows)
SEED = 42

def try_parse_list(x):
    """Parse stringified python lists safely."""
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def parse_turns(row):
    """
    Parse turns from transformed_conversation (JSON list of {turn, speaker, text}).
    Returns list of (text, speaker) tuples.
    """
    tc = row.get("transformed_conversation", "")
    if not isinstance(tc, str) or not tc.strip():
        return []
    try:
        objs = json.loads(tc)
        if isinstance(objs, list):
            return [(t.get("text", "").strip(), t.get("speaker", "")) for t in objs
                    if t.get("text", "").strip()]
    except Exception:
        pass
    return []

def pick_best_turn(row, turns):
    """
    Pick the most problematic turn index using turn_problematic_avg scores.
    Falls back to turn with most VC labels, then middle turn.
    """
    n = len(turns)
    if n == 0:
        return None, "no_turns"

    # 1) turn_problematic_avg: list of per-turn scores, same length as turns
    tpa = try_parse_list(row.get("turn_problematic_avg"))
    if isinstance(tpa, list) and len(tpa) == n:
        scores = []
        for v in tpa:
            try:
                scores.append(float(v))
            except Exception:
                scores.append(float("nan"))
        best_i = max(range(n), key=lambda i: (-1e9 if pd.isna(scores[i]) else scores[i]))
        return best_i, "turn_problematic_avg"

    # 2) turn_vc_union: list of VC label lists per turn
    vcu = try_parse_list(row.get("turn_vc_union"))
    if isinstance(vcu, list) and len(vcu) == n:
        counts = [len(e) if isinstance(e, (list, tuple, set)) else 0 for e in vcu]
        best_i = max(range(n), key=lambda i: counts[i])
        return best_i, "turn_vc_union"

    # fallback: middle turn
    return n // 2, "fallback"

def main():
    m = pd.read_csv(IN_PATH)

    # keep couples only
    m = m[m["relationship_subtype"] == "couple"].copy()

    # keep rows where condition exists
    m = m[m["condition"].notna()].copy()

    # sample ids that have both conditions (positive + negative)
    id_cond_counts = m.groupby("id")["condition"].nunique()
    paired_ids = id_cond_counts[id_cond_counts >= 2].index.tolist()

    if len(paired_ids) == 0:
        print("No paired ids found. Sampling from available rows.")
        candidates = m.sample(n=min(N_PER_CONDITION * 2, len(m)), random_state=SEED)
    else:
        import random
        random.seed(SEED)
        sample_ids = paired_ids[:]
        random.shuffle(sample_ids)
        sample_ids = sample_ids[:min(N_PER_CONDITION, len(sample_ids))]
        candidates = m[m["id"].isin(sample_ids)].copy()

    rows = []
    for _, r in candidates.iterrows():
        turns = parse_turns(r)          # (text, speaker) tuples from transformed_conversation
        ti, method = pick_best_turn(r, turns)
        if ti is None:
            continue
        seed_text, seed_speaker = turns[ti]

        # VC label count from dataset annotation (ground truth for original)
        vcu = try_parse_list(r.get("turn_vc_union"))
        vc_labels = []
        if isinstance(vcu, list) and ti < len(vcu) and isinstance(vcu[ti], list):
            vc_labels = vcu[ti]
        orig_vc_count = len(vc_labels)
        orig_vc_labels = "|".join(vc_labels) if vc_labels else ""

        rows.append({
            "id": r.get("id"),
            "condition": r.get("condition"),
            "relationship_subtype": r.get("relationship_subtype"),
            "relationship_tag": r.get("relationship_tag"),
            "backstory_used_in_mturk": r.get("backstory"),
            "positive_backstory": r.get("positive_backstory"),
            "negative_backstory": r.get("negative_backstory"),
            "seed_turn_index": ti,
            "seed_selection_method": method,
            "seed_speaker_guess": seed_speaker,
            "seed_utterance": seed_text,
            "orig_vc_count": orig_vc_count,
            "orig_vc_labels": orig_vc_labels,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    print("Saved ->", OUT_PATH)
    print("Rows saved:", len(out))
    print("Conditions in seeds:", out["condition"].value_counts().to_dict())
    print(out[["id","condition","seed_turn_index","seed_selection_method"]].head(10))

if __name__ == "__main__":
    main()