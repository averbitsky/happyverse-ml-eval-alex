# Interview Eval

AI-powered interview evaluation using **LangChain** and **GPT-4** for automated fact-checking, hallucination detection, structured scoring, and polished one-page visualizations.

---

## Capabilities & Quality Coverage

| Dimension            | Implementation Highlights                                                                 |
|---------------------|---------------------------------------------------------------------------------------------|
| GenTech Fluency     | Async LangChain chains; structured Pydantic outputs; prompt-driven flows                    |
| Rigor               | 3-axis scoring; web verification; 50+ criteria; deal-breaker logic                          |
| Modularity          | BaseEvaluator abstraction; YAML prompts; pluggable search/verifiers                         |
| Visualization       | Auto-generated single-page PDF with consistent layout and typography                        |
| Practicality        | Namespace-isolated caching; batch mode; configurable parallelism; clean JSON artifacts      |

---

## Highlights

- **Parallel evaluators** for Plausibility, Technical, and Communication (0–100).
- **Entity extraction & web verification** with a 1000+ tech knowledge base.
- **SQLite-backed cache** with namespace isolation per transcript.
- **One-page PDF** per candidate: scores, rationale, strengths/concerns/deal-breakers.

---

## Architecture (at a glance)

- **LangChain** orchestrates async chains.
- **OpenAI (GPT-4)** drives scoring, synthesis, and extraction.
- **DuckDuckGo Search** verifies entities/claims in real time (cached).
- **SQLite + in-memory LRU** provides persistent caching and metrics.

---

## What Problems This Solves

**Grounding & Retrieval**  
Extracts companies/technologies/claims and validates via search & local KB. Per-transcript cache namespaces prevent cross-candidate contamination.

**Verification & Hallucination Detection**  
1) Known-tech DB check (instant)  
2) Web search for unknowns (cached)  
3) LLM plausibility in context  
4) Cross-question consistency  
→ Flags **impossible_claims**, **missing_specifics**, contradictions.

**Scale Across Candidates/Jobs**  
Modular evaluators; YAML prompts (no code edits). Configurable parallelism and batch runs. Fresh evaluator instances per transcript for isolation.

---

## Evaluation Dimensions

- **Plausibility** — truthfulness & feasibility  
- **Technical** — accuracy & depth  
- **Communication** — clarity & professionalism  

**Synthesis** produces: Recommendation (Strong/Weak Yes/No), Confidence, Risk (Low/Medium/High), Deal-breakers, and detailed rationale.

---

## Quick Start

```bash
# Install
pip install -e .

# Configure (set your API key)
echo "OPENAI_API_KEY=sk-..." > .env

# Run
python src/run_evaluation.py
```

---

## Inputs

Place files in `data/`:
- `job_description.txt` — role requirements
- `questions.txt` — questions separated by **double newlines**
- `transcripts/*.txt` — Q&A style transcripts (supports `Q1:/A1:` or raw text)

---

## Outputs

- **Per-candidate JSON** + **PDF** visualization:
  - Horizontal bar chart with per-question markers
  - Color-coded decision & risk
  - Strengths / Concerns / Deal-breakers
  - Centered **Candidate Profile** rationale
- Optional **comparison PDF** when evaluating multiple candidates.

---

## CLI & Config

```bash
# Common options
python src/run_evaluation.py \
  --model gpt-4 \
  --parallel 5 \
  --no-cache \
  --clean-cache \
  --export-cache \
  --no-visualization
```

**Key environment settings** (via `.env` or env vars):
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (default: `gpt-4`)
- `MAX_PARALLEL_EVALUATIONS` (default: `3`)

---

## Project Structure

```python
src/intervieweval/
├── evaluators/       # Plausibility, Technical, Communication, Synthesis
├── tools/            # Search, entity verification
├── cache/            # Persistent cache (SQLite + LRU)
├── models/           # Pydantic schemas
├── prompts/          # YAML templates
└── visualization/    # PDF generation
```

---

## Extending the System

```python
# Add a new evaluator (example)
from intervieweval.evaluators.base import BaseEvaluator

class MedicalEvaluator(BaseEvaluator):
    def get_prompt_key(self) -> str:
        return "medical_assessment"

# Swap models at runtime
from intervieweval.config.settings import Settings
settings = Settings(openai_model="gpt-4o-mini")

# Use custom prompts
from intervieweval.prompts.manager import PromptManager
prompt_manager = PromptManager("custom_prompts.yaml")
```
