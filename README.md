# llm-conflict-deescalation-analysis
## An Exploratory Study of Harmful Language Patterns in Interpersonal Conflict

### Motivation

Interpersonal conflicts often escalate due to harmful discourse patterns such as absolutist language ("you always"), blame framing, contempt markers, and emotional invalidation. While large language models are capable of rewriting text, it remains unclear whether they systematically reduce such harmful markers and promote more constructive communication.

This project extends an emotion-aware communication assistant into a small-scale exploratory empirical study.

### Research Question

Do LLM-generated rewrites reduce measurable harmful discourse markers in real-world conflict discussions?

---

### Method

**Dataset**
- 30 real-world conflict discussion excerpts collected from public online forums.

**Harmful Marker Taxonomy**
- Absolutist language (e.g., "always", "never")
- Blame framing ("you did", "you don't")
- Contempt cues ("whatever", "fine.")
- Mind-reading language ("you think", "you just")

**Procedure**
1. Count harmful markers in original messages.
2. Generate empathetic rewrites using a structured LLM prompt.
3. Recount harmful markers in rewritten messages.
4. Compare aggregate statistics.

---

### Preliminary Findings

- Mean harmful markers per message decreased after rewrite.
- Absolutist language was substantially reduced.
- Rewrites shifted from accusation framing to feeling/need framing.

(Full quantitative breakdown in notebooks/)

---

### Technical Implementation

- LLM: Claude Sonnet API
- Structured JSON output enforcement
- Python-based marker analysis
- Exploratory statistical comparison

---

### Future Directions

- Human annotation for validation
- Inter-rater agreement analysis
- Discourse-level modeling of emotional reframing
- Cross-cultural variation study