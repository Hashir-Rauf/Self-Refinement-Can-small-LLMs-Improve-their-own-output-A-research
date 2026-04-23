# Self-Refinement: Can Small LLMs Improve Their Own Output?

**Course:** CS4063 - Natural Language Processing  
**Section:** BDS-6B  
**Team:** Hashir Rauf (23L-2572), Bilal Ahmad (23L-2534), Muhammad Saad (23L-2620)

---

## Overview

This project empirically evaluates whether small language models (2B-7B parameters) can improve their own outputs through iterative self-refinement. Rather than training or fine-tuning, all experiments use prompt engineering against locally hosted models via Ollama.

The study compares six conditions on text summarization using the CNN/DailyMail dataset and introduces a diversity metric to detect refinement collapse, a failure mode where successive iterations converge to nearly identical outputs despite feedback.

---

## Experimental Conditions

| ID | Method | Description |
|----|--------|-------------|
| B1 | Zero-shot | Single-pass baseline generation |
| B2 | Chain-of-Thought (CoT) | Explicit step-by-step reasoning before output |
| B3 | Rephrase-and-Respond (RaR) | Task rephrasing before generation |
| SR-1 | Self-Refine k=1 | Generate, get feedback, refine once |
| SR-2 | Self-Refine k=2 | Generate, get feedback, refine twice |
| SR-3 | Self-Refine k=3 | Generate, get feedback, refine three times |

---

## Models

| Model | Parameters | Notes |
|-------|-----------|-------|
| `gemma4:e4b` | ~2B (e4-quantized) | Primary model; produces usable summaries |
| `mistral:7b` | 7B | Secondary model; collapses on summarization |

Both models are served locally through Ollama and queried via its REST API. No GPU fine-tuning or transformers inference is used.

---

## Repository Structure

```
.
├── implementation.ipynb          # Full implementation (single notebook)
├── data/
│   ├── cnn_samples.json          # 20 CNN/DailyMail articles (test set)
│   └── squad_samples.json        # 20 SQuAD QA pairs (loaded, not used in main experiments)
├── results/
│   ├── metrics.csv               # Aggregated metrics (240 records)
│   ├── fig1_performance_by_condition.png
│   ├── fig2_iteration_rouge.png
│   ├── fig3_diversity_collapse.png
│   ├── fig4_heatmap_gemma4_e4b.png
│   ├── fig4_heatmap_mistral_7b.png
│   └── raw_summarize_*.json      # 12 raw result files (model x condition)
└── Combined_Analysis_All_10_Papers.docx  # Literature survey
```

---

## Requirements

**Python 3.10+** with the following packages:

```
requests
rouge-score
bert-score
datasets
numpy
pandas
matplotlib
seaborn
sentence-transformers
tqdm
```

Install all at once:

```bash
pip install requests rouge-score bert-score datasets numpy pandas matplotlib seaborn sentence-transformers tqdm
```

**Ollama** must be installed and running locally. Pull the required models:

```bash
ollama pull gemma4:e4b
ollama pull mistral:7b
```

---

## Running the Project

1. Start the Ollama server:

   ```bash
   ollama serve
   ```

2. Launch the notebook:

   ```bash
   jupyter notebook implementation.ipynb
   ```

3. Run cells in order. The notebook is divided into sections:

   | Section | Content |
   |---------|---------|
   | 0 | Package installation |
   | 1 | Configuration |
   | 2 | Ollama connection test |
   | 3 | Dataset loading |
   | 4 | Prompt templates |
   | 5 | Experiment runner |
   | 6 | Run all conditions (2-5 hours for 20 samples x 2 models) |
   | 7 | Metric computation (ROUGE, BERTScore, diversity) |
   | 8-12 | Analysis, ablations, hypothesis testing |

Results are checkpointed after each condition to `results/raw_summarize_*.json`, so the experiment can be safely interrupted and resumed.

---

## Configuration

Key parameters in Section 1 of the notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `PRIMARY_MODEL` | `gemma4:e4b` | Main model for experiments |
| `SECONDARY_MODEL` | `mistral:7b` | Comparison model |
| `N_SAMPLES` | `20` | Samples per task |
| `MAX_ITERATIONS` | `3` | Max self-refinement iterations |
| `TEMPERATURE` | `0.7` | Generation temperature |
| `FEEDBACK_TEMP` | `0.3` | Feedback generation temperature |
| `MAX_ARTICLE_WORDS` | `300` | Article truncation limit |
| `MAX_TOKENS_GEN` | `256` | Max tokens for generation |
| `MAX_TOKENS_FB` | `150` | Max tokens for feedback |
| `SEED` | `42` | Random seed |

---

## Evaluation Metrics

- **ROUGE-1, ROUGE-2, ROUGE-L** - lexical overlap with reference summaries
- **BERTScore F1** - semantic similarity via `all-MiniLM-L6-v2` embeddings
- **Diversity Score** - average cosine distance between consecutive refinement outputs; values below 0.03 indicate collapse

---

## Key Results

**Gemma4:e4b (primary model):**

| Condition | ROUGE-1 |
|-----------|---------|
| Zero-shot | 0.3055 |
| CoT | 0.2099 |
| RaR | 0.2876 |
| SR-1 | 0.3011 |
| SR-2 | 0.3132 (best) |
| SR-3 | 0.2950 |

Self-refinement at k=2 yields the best ROUGE-1, but the gain over zero-shot is marginal (+2.5%). Performance degrades at k=3, suggesting over-refinement.

**Mistral 7B:** Complete failure across all conditions (ROUGE-1 ~0.013). The model produces generic single-token outputs regardless of prompt, making self-refinement ineffective.

---

## Hypotheses and Findings

| Hypothesis | Result |
|------------|--------|
| H1: Self-refine shows no improvement over baseline | Supported. SR-3 approximates zero-shot performance. |
| H2: Mistral degrades earlier than Gemma | Not supported. Mistral fails on all conditions, not gradually. |
| H3: Simple prompt engineering outperforms self-refine | Partially supported. RaR underperforms zero-shot but avoids refinement overhead. |
| H4: Diversity collapse is a significant failure mode | Supported. 5% of Gemma4:e4b runs show collapse (diversity < 0.03). |

---

## Datasets

**CNN/DailyMail** (`data/cnn_samples.json`)
- 20 articles sampled from the test split
- Articles truncated to 300 words to fit small model context windows
- Source: HuggingFace `cnn_dailymail` v3.0.0

**SQuAD** (`data/squad_samples.json`)
- 20 QA pairs from the validation split
- Contexts truncated to 150 words
- Source: HuggingFace `squad`
- Loaded but not used in the main experiment runs
