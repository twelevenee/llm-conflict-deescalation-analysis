import os
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

IN_PATH = "data/processed/mturk_seeds.csv"
OUT_PATH = "data/processed/rewrites.csv"

MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.2
MAX_TOKENS = 200

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM = "You are a careful assistant that rewrites messages using Nonviolent Communication (NVC)."

def make_prompt(backstory: str | None, utterance: str) -> str:
    if backstory:
        return f"""Task: Rewrite the message using Nonviolent Communication (NVC), considering the relationship backstory.

Relationship backstory:
{backstory}

Original utterance:
{utterance}

Requirements:
- Preserve the core intent.
- Remove blame, moral judgment, absolutist language, and demands.
- Express (when possible): (1) observation (2) feeling (3) need (4) request.
- Keep it to 1–2 sentences.
- Do NOT add new facts beyond what is in the utterance and backstory.
Return only the rewritten text."""
    else:
        return f"""Task: Rewrite the message using Nonviolent Communication (NVC).

Original utterance:
{utterance}

Requirements:
- Preserve the core intent.
- Remove blame, moral judgment, absolutist language, and demands.
- Express (when possible): (1) observation (2) feeling (3) need (4) request.
- Keep it to 1–2 sentences.
Return only the rewritten text."""

def call_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        ),
    )
    return response.text.strip()

def main():
    df = pd.read_csv(IN_PATH)

    # output file exists? resume-friendly
    if os.path.exists(OUT_PATH):
        out = pd.read_csv(OUT_PATH)
        done = set(zip(out["id"], out["condition"]))
        print(f"Found existing {OUT_PATH} with {len(out)} rows. Will skip done rows.")
    else:
        out = pd.DataFrame()
        done = set()

    rows = []
    for _, r in df.iterrows():
        key = (r["id"], r["condition"])
        if key in done:
            continue

        utt = str(r["seed_utterance"]).strip()
        pos_bs = str(r["positive_backstory"]).strip()
        neg_bs = str(r["negative_backstory"]).strip()

        # 3 conditions we want for EACH row:
        # - none (no backstory)
        # - pos
        # - neg
        prompt_none = make_prompt(None, utt)
        prompt_pos  = make_prompt(pos_bs, utt)
        prompt_neg  = make_prompt(neg_bs, utt)

        rewrite_none = call_gemini(prompt_none)
        rewrite_pos  = call_gemini(prompt_pos)
        rewrite_neg  = call_gemini(prompt_neg)

        rows.append({
            "id": r["id"],
            "condition": r["condition"],  # MTurk condition for this row (positive/negative)
            "seed_utterance": utt,
            "positive_backstory": pos_bs,
            "negative_backstory": neg_bs,
            "rewrite_none": rewrite_none,
            "rewrite_pos": rewrite_pos,
            "rewrite_neg": rewrite_neg,
            "model": MODEL,
            "temperature": TEMPERATURE,
        })

        # save incrementally (crash-safe)
        tmp = pd.DataFrame(rows)
        if not out.empty:
            merged = pd.concat([out, tmp], ignore_index=True)
        else:
            merged = tmp
        merged.to_csv(OUT_PATH, index=False)

        print(f"✓ saved row id={r['id']} condition={r['condition']} -> {OUT_PATH}")

    # final save
    if rows:
        final = pd.concat([out, pd.DataFrame(rows)], ignore_index=True) if not out.empty else pd.DataFrame(rows)
        final.to_csv(OUT_PATH, index=False)
        print(f"Done. Total rows in rewrites: {len(final)}")
    else:
        print("Nothing new to generate (all rows already done).")

if __name__ == "__main__":
    main()