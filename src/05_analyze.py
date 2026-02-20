"""
Compares harmful markers: original (dataset VC labels) vs. rewrites (lexical).
Requires: data/processed/claude_outputs_parsed.csv
Run after: python3 src/04_parse_claude_outputs.py
"""
import re
import pandas as pd

IN_PATH   = "data/processed/claude_outputs_parsed.csv"
SEED_PATH = "data/processed/mturk_seeds_10ids.csv"

# Expanded lexical taxonomy aligned with VC types in PersonaConflicts
MARKERS = {
    # Absolutist / Overgeneralization
    "absolutist": re.compile(
        r"\b(always|never|every time|all the time|constantly|forever|"
        r"nothing|everything|everyone|nobody|no one)\b", re.I),
    # Blame / Accusation
    "blame": re.compile(
        r"\b(you did|you don'?t|you never|you always|your fault|you made|"
        r"because of you|you caused|you ruined|you broke|you lost)\b", re.I),
    # Contempt / Dismissal
    "contempt": re.compile(
        r"\b(whatever|i don'?t care|forget it|seriously\?|obviously|ridiculous|"
        r"how on earth|what on earth|good luck with that|as if|yeah right)\b"
        r"|fine\.", re.I),
    # Mind-reading / Assumption
    "mind_reading": re.compile(
        r"\b(you think|you just|you only|you want|you feel like|you don'?t care|"
        r"you clearly|you obviously|you never care|you always think)\b", re.I),
    # Moralistic judgment
    "moralistic": re.compile(
        r"\b(you should(n'?t)?|you need to|you must|you have to|"
        r"you ought to|that'?s wrong|that'?s bad|irresponsible|selfish|"
        r"makes you the expert|your fair share of)\b", re.I),
    # Demands
    "demands": re.compile(
        r"\b(won'?t you|you better|you will|do it now|"
        r"this won'?t sort itself|just do it)\b", re.I),
    # Sarcasm markers
    "sarcasm": re.compile(
        r"\b(huh,|I guess (that|you)|magic thing|how do you manage|"
        r"congratulations on|great job on)\b", re.I),
}

MARKER_COLS = [f"marker_{k}" for k in MARKERS] + ["marker_total"]


def count_markers(text: str) -> dict:
    counts = {k: len(pat.findall(str(text))) for k, pat in MARKERS.items()}
    counts["total"] = sum(counts.values())
    return counts


def main():
    df    = pd.read_csv(IN_PATH)
    seeds = pd.read_csv(SEED_PATH)
    seeds = seeds.rename(columns={"id": "row_id", "condition": "mturk_condition"})

    merged = df.merge(
        seeds[["row_id", "mturk_condition", "seed_utterance",
               "orig_vc_count", "orig_vc_labels"]],
        on=["row_id", "mturk_condition"], how="left"
    )

    print("=== Original utterances: VC label counts (dataset annotation) ===")
    orig = seeds.drop_duplicates("row_id")
    print(f"  mean VC count  = {orig['orig_vc_count'].mean():.3f}")
    print(f"  max  VC count  = {orig['orig_vc_count'].max()}")
    print(f"\n  VC label distribution:")
    all_labels = [lb for labels in orig["orig_vc_labels"].dropna()
                  for lb in labels.split("|") if lb]
    label_counts = pd.Series(all_labels).value_counts()
    print(label_counts.to_string())

    NVC_COMPONENTS = ["observation", "feeling", "need", "request", "empathy"]
    nvc_cols = [f"nvc_{c}" for c in NVC_COMPONENTS] + ["nvc_total"]

    print("\n=== Rewrite: lexical harmful markers by backstory condition ===")
    for k, pat in MARKERS.items():
        df[f"marker_{k}"] = df["rewrite"].apply(lambda x: len(pat.findall(str(x))))
    df["marker_total"] = df[[f"marker_{k}" for k in MARKERS]].sum(axis=1)
    merged = df.merge(
        seeds[["row_id", "mturk_condition", "seed_utterance",
               "orig_vc_count", "orig_vc_labels"]],
        on=["row_id", "mturk_condition"], how="left"
    )
    print(merged.groupby("backstory_condition")[MARKER_COLS].mean().round(3))

    print("\n=== Key comparison: orig VC count vs rewrite marker total ===")
    summary = merged.groupby("backstory_condition").agg(
        orig_vc_mean=("orig_vc_count", "mean"),
        rewrite_marker_mean=("marker_total", "mean"),
    ).round(3)
    summary["reduction"] = (summary["orig_vc_mean"] - summary["rewrite_marker_mean"]).round(3)
    print(summary)

    # NVC component analysis (only if annotation data present)
    if "nvc_total" in df.columns:
        print("\n=== NVC component presence by backstory condition ===")
        print("(1 = present in rewrite, mean across items)")
        print(merged.groupby("backstory_condition")[nvc_cols].mean().round(3))

        print("\n=== Hypothesis: does backstory increase empathy/feeling/need? ===")
        pivot_nvc = merged.pivot_table(
            index=["row_id", "mturk_condition"],
            columns="backstory_condition",
            values=nvc_cols
        )
        for comp in ["nvc_empathy", "nvc_feeling", "nvc_need"]:
            cols_flat = [c for c in pivot_nvc.columns if c[0] == comp]
            if len(cols_flat) < 2:
                continue
            sub = pivot_nvc[[c for c in pivot_nvc.columns if c[0] == comp]]
            sub.columns = [c[1] for c in sub.columns]
            for cond in ["pos", "neg"]:
                if cond in sub.columns and "none" in sub.columns:
                    delta = (sub[cond] - sub["none"]).mean()
                    print(f"  {comp:15s}  {cond} − none = {delta:+.3f}")

    print("\n=== Per-row detail (backstory=none) ===")
    base_cols = ["row_id", "orig_vc_count", "orig_vc_labels", "marker_total"]
    nvc_show  = [c for c in nvc_cols if c in merged.columns]
    none_rows = merged[merged["backstory_condition"] == "none"][
        base_cols + nvc_show + ["rewrite"]
    ].copy()
    none_rows["rewrite"] = none_rows["rewrite"].str[:60] + "…"
    print(none_rows.sort_values("orig_vc_count", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
