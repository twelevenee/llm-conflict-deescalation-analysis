# LLM-Guided Discourse De-escalation
### Backstory-Conditioned NVC Rewriting of Interpersonal Conflict Utterances

An exploratory NLP study investigating whether LLM-generated rewrites reduce harmful discourse markers in interpersonal conflict dialogues — and whether relationship backstory conditions the quality of those rewrites.

---

## Research Questions

1. **RQ1**: Do LLM-generated NVC rewrites reduce harmful discourse markers relative to original conflict utterances?
2. **RQ2**: Does providing relationship backstory (positive vs. negative relational history) condition the NVC component structure of rewrites?

---

## Dataset

**PersonaConflicts Corpus** (Shen et al., EMNLP 2025)
5,772 simulated interpersonal conflict dialogues with rich relational backstories (romantic partners, family, friends). Each dialogue includes crowdsourced turn-level annotations for Violent Communication (VC) types (Moralistic Judgment, Demand, Comparison, Denial of Responsibility, Deserve Thinking) and NVC types (Feeling Statement, Need Statement, etc.).

This study uses a subset of **couple-relationship dialogues** (n = 10 dialogue IDs × 2 MTurk backstory conditions = 20 seed utterances).

> Shen et al. (2025). *Words Like Knives: Backstory-Personalized Modeling and Detection of Violent Communication.* EMNLP 2025.

*Raw data not included in this repository. Obtain from the original corpus release.*

---

## Method

### Experimental Design

A **3-condition within-subjects design** applied to each seed utterance:

| Condition | Backstory provided to LLM |
|-----------|--------------------------|
| `none`    | No backstory |
| `pos`     | Positive relational history |
| `neg`     | Negative relational history |

### Seed Selection

High-conflict utterances selected per dialogue using `turn_problematic_avg` scores (MTurk crowdsourced ratings), with fallback to `turn_vc_union` label density. Source: `transformed_conversation` JSON field in the corpus.

### Rewrite Generation

Claude (claude.ai) prompted with structured NVC rewrite instructions + backstory context. JSON-format output enforced for both the rewrite and component annotation.

### Annotation

**Original utterances**: VC label counts from corpus annotation (expert ground truth)
**Rewrites**: LLM self-annotation of NVC component presence (observation / feeling / need / request / empathy)

---

## Results

### RQ1: VC Marker Reduction

| Condition | Orig. VC (mean) | Rewrite markers (mean) | Reduction |
|-----------|----------------|----------------------|-----------|
| none      | 1.15           | 0.00                 | **1.15**  |
| pos       | 1.15           | 0.00                 | **1.15**  |
| neg       | 1.15           | 0.05                 | **1.10**  |

LLM rewriting consistently eliminated surface-level VC markers across all backstory conditions. Original VC types present: Moralistic Judgment (4), Demand (3), Comparison (2), Denial of Responsibility (1), Deserve Thinking (1).

### RQ2: NVC Component Structure by Backstory Condition

| Component   | none | pos  | neg  | neg − none |
|-------------|------|------|------|-----------|
| observation | 0.40 | 0.40 | 0.40 | 0.00      |
| feeling     | 0.60 | 0.60 | **0.70** | **+0.10** |
| need        | 0.85 | 0.80 | 0.80 | −0.05     |
| request     | 0.55 | 0.60 | 0.55 | 0.00      |
| empathy     | 0.30 | 0.30 | 0.30 | 0.00      |
| **total**   | 2.85 | 2.95 | **3.05** | **+0.20** |

Negative backstory conditioning was associated with higher feeling expression rates (+0.10), suggesting backstory influences emotional register even when surface VC markers are uniformly eliminated.

### Key Interpretation

VC marker removal shows a **ceiling effect** — the LLM eliminates harmful markers regardless of backstory — but NVC component analysis reveals that backstory conditioning operates at a **deeper pragmatic level**, shifting the emotional tenor of the rewrite. This supports a **mechanism hypothesis**: backstory does not affect whether de-escalation occurs, but *how* it is expressed.

---

## Limitations

- Small exploratory sample (n = 10 dialogue IDs)
- Measurement asymmetry: original VC uses expert annotation; rewrite quality uses LLM self-annotation
- Some seed utterances have low baseline conflict intensity (VC count = 0)
- Single LLM annotator; no inter-rater reliability
- The +0.10 feeling effect requires replication at larger scale

---

## Pipeline

```
data/raw/mturk_aggregate.csv          (PersonaConflicts corpus, not included)
         │
         ▼
src/01_inspect_dataset.py             Explore raw data structure
src/01_inspect_mturk.py               Explore MTurk annotation structure
         │
         ▼
src/02_build_mturk_seeds.py           Select high-conflict seed utterances
         │                            → data/processed/mturk_seeds.csv
         ▼
src/06_generate_batch_prompt.py       Generate structured batch prompt
         │                            → data/processed/batch_prompt.txt
         │
         ▼  [Paste into Claude.ai — see below]
         │
         ▼
data/processed/claude_outputs.jsonl   LLM rewrites + NVC annotations
         │
         ▼
src/04_parse_claude_outputs.py        Parse JSON → CSV + lexical scoring
         │                            → data/processed/claude_outputs_parsed.csv
         ▼
src/05_analyze.py                     Comparative analysis + results
```

### Running the Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Inspect data
python src/01_inspect_dataset.py
python src/01_inspect_mturk.py

# 3. Build seeds
python src/02_build_mturk_seeds.py

# 4. Generate batch prompt
python src/06_generate_batch_prompt.py
# → Open data/processed/batch_prompt.txt, paste into claude.ai
# → Copy JSON response → save as data/processed/claude_outputs.jsonl

# 5. Parse and analyze
python src/04_parse_claude_outputs.py
python src/05_analyze.py
```

*`src/03_generate_rewrites.py` is an alternative API-based pipeline (Anthropic/Gemini) for automated rewriting without manual web interaction.*

---

## Relation to Prior Work

The PersonaConflicts corpus frames backstory as a conditioning variable for **conflict perception** (how harmful a turn is judged to be). This project shifts the focus to **conflict transformation**: does the same backstory variable shape how an LLM de-escalates the conflict? The null finding on VC removal — combined with the positive signal on feeling expression — suggests these two functions (perception vs. transformation) may be differentially sensitive to relational context.

---

## Future Directions

- Scale to full 40-utterance seed set and larger backstory sample
- Human evaluation of de-escalation quality (perceived empathy, authenticity)
- Replace LLM self-annotation with independent annotator or fine-tuned classifier
- Cross-relationship-type comparison (couple vs. sibling vs. parent-child)
- Lexical diversity and hedging frequency as additional rewrite quality metrics

---

## Stack

- Python (pandas, re)
- Claude (claude.ai, structured JSON prompting)
- PersonaConflicts corpus (Shen et al., EMNLP 2025)
