"""
Parses claude_outputs.jsonl (JSON array or concatenated objects) into a CSV,
then counts harmful markers lexically.

Run after saving Claude's response to data/processed/claude_outputs.jsonl.
"""
import json
import re
import pandas as pd
from pathlib import Path

IN_PATH  = Path("data/processed/claude_outputs.jsonl")
OUT_PATH = Path("data/processed/claude_outputs_parsed.csv")

# Lexical taxonomy (from README)
MARKERS = {
    "absolutist":  re.compile(
        r"\b(always|never|every time|all the time|constantly|forever|nothing|everything)\b",
        re.IGNORECASE),
    "blame":       re.compile(
        r"\b(you did|you don'?t|you never|you always|your fault|you made|because of you)\b",
        re.IGNORECASE),
    "contempt":    re.compile(
        r"\b(whatever|fine\.|i don'?t care|forget it|seriously\?|obviously|ridiculous)\b",
        re.IGNORECASE),
    "mind_reading": re.compile(
        r"\b(you think|you just|you only|you want|you feel like|you don'?t care)\b",
        re.IGNORECASE),
}


def count_markers(text: str) -> dict:
    return {k: len(pat.findall(text)) for k, pat in MARKERS.items()}


def iter_json_objects(text: str):
    """Parse a JSON array or concatenated JSON objects robustly."""
    text = text.strip()
    # Case 1: it's a JSON array
    if text.startswith("["):
        try:
            objs = json.loads(text)
            yield from objs
            return
        except json.JSONDecodeError:
            pass
    # Case 2: concatenated / pretty-printed objects
    buf, depth, in_str, esc = "", 0, False, False
    for ch in text:
        buf += ch
        if in_str:
            if esc:    esc = False
            elif ch == "\\": esc = True
            elif ch == '"':  in_str = False
            continue
        if ch == '"':  in_str = True; continue
        if ch == "{":  depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                s = buf.strip(); buf = ""
                if s:
                    yield json.loads(s)


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {IN_PATH}.\n"
            "Paste Claude's JSON response there, then re-run.")

    text = IN_PATH.read_text(encoding="utf-8").strip()
    if not text:
        print("claude_outputs.jsonl is empty — paste Claude's response first.")
        return

    objs = list(iter_json_objects(text))
    if not objs:
        print("No JSON objects found. Check the file contents.")
        return

    NVC_COMPONENTS = ["observation", "feeling", "need", "request", "empathy"]

    rows = []
    for obj in objs:
        rewrite = obj.get("rewrite", "")
        markers = count_markers(rewrite)
        nvc = obj.get("nvc", {})
        nvc_row = {c: int(bool(nvc.get(c, {}).get("present", False)))
                   for c in NVC_COMPONENTS}
        rows.append({
            "row_id":              obj.get("row_id"),
            "mturk_condition":     obj.get("mturk_condition"),
            "backstory_condition": obj.get("backstory_condition"),
            "rewrite":             rewrite,
            **{f"marker_{k}": v for k, v in markers.items()},
            "marker_total":        sum(markers.values()),
            **{f"nvc_{c}": v for c, v in nvc_row.items()},
            "nvc_total":           sum(nvc_row.values()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH} ({len(df)} rows)")
    print("\nMean harmful marker count by backstory condition:")
    print(df.groupby("backstory_condition")["marker_total"].mean().round(2))


if __name__ == "__main__":
    main()
