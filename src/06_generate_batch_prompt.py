"""
Generates a single batch prompt file for all 20 rows × 3 conditions (60 rewrites).
Paste the output file into Claude.ai, save the response as
data/processed/claude_outputs.jsonl, then run:
  python3 src/04_parse_claude_outputs.py
"""
import json
import pandas as pd
from pathlib import Path

IN_PATH  = Path("data/processed/mturk_seeds_10ids.csv")
OUT_PATH = Path("data/processed/batch_prompt.txt")


def main():
    df = pd.read_csv(IN_PATH)

    items = []
    for _, r in df.iterrows():
        utt     = str(r["seed_utterance"]).strip()
        pos_bs  = str(r["positive_backstory"]).strip()
        neg_bs  = str(r["negative_backstory"]).strip()
        row_id  = int(r["id"])
        mturk_c = str(r["condition"])

        items.append({"row_id": row_id, "mturk_condition": mturk_c,
                      "backstory_condition": "none", "backstory": "", "utterance": utt})
        items.append({"row_id": row_id, "mturk_condition": mturk_c,
                      "backstory_condition": "pos",  "backstory": pos_bs, "utterance": utt})
        items.append({"row_id": row_id, "mturk_condition": mturk_c,
                      "backstory_condition": "neg",  "backstory": neg_bs, "utterance": utt})

    prompt = f"""You are a careful assistant that rewrites conflict utterances using Nonviolent Communication (NVC) and annotates the result.

Below are {len(items)} items. Each has:
- row_id: integer ID
- mturk_condition: the original MTurk experimental condition ("positive" or "negative")
- backstory_condition: "none" | "pos" | "neg"
- backstory: relationship backstory (empty string if none)
- utterance: the original conflict utterance to rewrite

Task A — Rewrite:
- Preserve the core intent.
- Remove blame, moral judgment, absolutist language, and demands.
- Express when possible: (1) observation (2) feeling (3) need (4) request.
- Keep it to 1–2 sentences.
- Do NOT add new facts beyond what is in the utterance and backstory.

Task B — Annotate the REWRITE (not the original) for NVC component presence:
- observation: factual description of what happened, without evaluation
- feeling: explicit emotional state of the speaker
- need: underlying value or need being expressed
- request: a concrete, doable ask (not a demand)
- empathy: acknowledgment of the other person's feelings or perspective

Rules: set present=true only if the component is clearly expressed in the rewrite.

Return ONLY a JSON array of {len(items)} objects — no markdown fences, no explanation.
Each object must have EXACTLY these keys:

{{
  "row_id": <int>,
  "mturk_condition": "<str>",
  "backstory_condition": "<str>",
  "rewrite": "<str>",
  "nvc": {{
    "observation": {{"present": <bool>}},
    "feeling":     {{"present": <bool>}},
    "need":        {{"present": <bool>}},
    "request":     {{"present": <bool>}},
    "empathy":     {{"present": <bool>}}
  }}
}}

Items:
{json.dumps(items, ensure_ascii=False, indent=2)}"""

    OUT_PATH.write_text(prompt, encoding="utf-8")
    print(f"Written: {OUT_PATH}")
    print(f"  {len(items)} items, {len(prompt):,} chars (~{len(prompt)//4:,} tokens)")
    print()
    print("Next steps:")
    print("  1. Open data/processed/batch_prompt.txt and copy ALL contents")
    print("  2. Paste into claude.ai → send")
    print("  3. Copy the JSON array response → save as data/processed/claude_outputs.jsonl")
    print("  4. python3 src/04_parse_claude_outputs.py")
    print("  5. python3 src/05_analyze.py")


if __name__ == "__main__":
    main()
