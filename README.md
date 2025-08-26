# Interview Eval

AI-powered interview evaluation using **LangChain** agents and **GPT-4** for automated fact-checking, hallucination detection, and professional visualizations.

## Core Architecture

**LangChain** orchestrates parallel evaluation chains. **GPT-4** powers assessment and entity extraction. **DuckDuckGo API** verifies claims in real-time. **SQLite** provides persistent caching with namespace isolation.

## How It Solves Key Challenges

### Retrieval and Grounding

The system extracts entities (companies, technologies, claims) from responses and verifies them through web search. A 1000+ technology knowledge base provides instant validation. Each transcript gets isolated cache namespace to prevent contamination.

### Factual Verification & Hallucination Detection

Four-stage verification pipeline:
1. Check against known technology database (instant)
2. Web search for unknown entities (cached)
3. LLM plausibility analysis in context
4. Cross-question consistency validation

The system flags impossible claims, missing specifics, and contradictions as red flags.

### Scaling Across Candidates/Jobs

- **Modular evaluators** - Independent chains for each dimension
- **Parallel processing** - Configurable 1-10 concurrent threads
- **Batch mode** - Fresh evaluator instances per transcript
- **YAML prompts** - Adapt to any job/questions without code changes

### Edge Case Handling

| Edge Case | Detection | Response |
|-----------|-----------|----------|
| Lies | Web verification + consistency | Flags as impossible_claims |
| Vagueness | Specificity analysis | Lists missing_specifics |
| Tangents | Relevance scoring | Tracks directness_score |
| Buzzwords | Depth analysis | Shallow vs deep classification |

## Evaluation Dimensions

Three LLM chains score 0-100:
- **Plausibility**: Truthfulness and feasibility
- **Technical**: Accuracy and depth
- **Communication**: Clarity and professionalism

Synthesis produces:
- Recommendation (Strong Yes/No, Weak Yes/No)
- Confidence (0-100%)
- Risk level (Low/Medium/High)
- Deal-breakers

## Quick Start

```bash
# Install
pip install -e .

# Configure
echo "OPENAI_API_KEY=sk-..." > .env

# Run
python src/run_evaluation.py
```

## Input Format

Place in `data/` directory:
- `job_description.txt` - Role requirements
- `questions.txt` - Questions separated by double newlines
- `transcripts/*.txt` - Q&A format transcripts

## Output

Professional one-page PDF per candidate with:
- Visual score breakdown by question
- Color-coded recommendation
- Prioritized strengths/concerns/deal-breakers
- Detailed rationale

## Addressing Evaluation Rubric

| Dimension | Implementation |
|-----------|---------------|
| **GenTech Fluency** | LangChain async chains, parallel agents, structured Pydantic outputs |
| **Rigor** | 3-dimensional scoring, web verification, 50+ criteria, deal-breaker detection |
| **Modularity** | BaseEvaluator abstraction, YAML prompts, pluggable components |
| **Visualization** | Auto-generated PDF with charts and actionable insights |
| **Creativity** | Entity extraction, consistency matrix, confidence scoring, intelligent caching |

## Features

### Prompt Engineering Innovation
- **Claim decomposition** into verifiable atoms
- **Counterfactual reasoning** ("If true, what evidence would exist?")
- **Consistency matrix** cross-references all responses

### Intelligent Caching
- Namespace isolation per transcript
- Differential TTL (1 hour for success, 5 min for errors)
- LRU memory cache with SQLite persistence

### Extensibility

```python
# Custom evaluator for new domains
class MedicalEvaluator(BaseEvaluator):
   def get_prompt_key(self): 
       return "medical_assessment"

# Swap models
settings = Settings(openai_model="gpt-3.5-turbo")

# Custom prompts
prompt_manager = PromptManager("custom_prompts.yaml")
```

## Project Structure

```python
src/intervieweval/
├── evaluators/       # Evaluation chains (plausibility, technical, communication)
├── tools/            # Web search and entity verification
├── cache/            # Persistent caching with LRU
├── models/           # Pydantic structured outputs
├── prompts/          # YAML template management
└── visualization/    # PDF generation
```

## Performance

- Single evaluation: 15-30 seconds (GPT-4)
- Batch of 5: 60-90 seconds (parallelized)
- Cache hit rate: 60-80% after warmup

## Configuration

```bash
# Command line options
python src/run_evaluation.py \
 --model gpt-4 \
 --parallel 5 \
 --no-cache \
 --clean-cache \
 --export-cache
```