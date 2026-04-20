# Implementation Plan: Self-Refinement in Small LLMs via Ollama

**Project:** Self-Refinement: Can Small LLMs Improve Their Own Output?  
**Group:** Hashir Rauf (23L-2572), Bilal Ahmad (23L-2534), Muhammad Saad (23L-2620) | BDS-6B  
**Course:** CS4063 – Natural Language Processing  
**Code Deadline:** 19th April 2026

---

## Research Question (from assignment)

> Can small LLMs improve their own output through iterative self-refinement — and if so, under what conditions does it help vs. hurt?

This is grounded directly in the literature gap identified across all 10 papers: **existing self-refinement work (Self-Refine, Self-Polish, Reflexion, SSR) exclusively tests on large proprietary models (GPT-3.5, GPT-4, PaLM-540B). No paper systematically studies these methods on sub-10B open-source models.**

---

## Model Choice (Ollama)

Run on the device where Ollama is installed. Recommended models (pick one or compare two):

| Model | Size | Why |
|---|---|---|
| `llama3.2:3b` | 3B params | Tiny — should show refinement limits |
| `phi3:mini` (3.8B) | 3.8B params | Strong reasoning for size |
| `mistral:7b` | 7B params | Standard benchmark |
| `gemma3:4b` | 4B params | Balanced |

**Recommended:** Use `llama3.2:3b` as primary + `mistral:7b` as secondary. This gives a small vs. medium comparison, directly addressing the scale question from the literature.

---

## Task Selection

Three tasks that are well-suited for measuring iterative refinement on small models:

| Task | Why Chosen | Metric |
|---|---|---|
| **Text Summarization** | ROUGE is objective; easy to detect quality drift | ROUGE-1, ROUGE-L |
| **Open-ended QA** | Tests factual coherence; shows hallucination drift | BERTScore, human eval |
| **Constrained Generation** | Forces specific output structure; clear pass/fail | Constraint satisfaction rate |

Start with **summarization** (primary task, fully automated metrics) and add QA as secondary.

---

## Dataset

- **CNN/DailyMail** (subset of 50–100 articles) for summarization — standard benchmark, has reference summaries for ROUGE
- **TriviaQA** or **SQuAD** (50 questions) for QA
- Use `datasets` library (HuggingFace) for loading

Reason for small subset: running locally on Ollama with iterative calls; 50–100 samples × 3 iterations × 2 models = ~600 total LLM calls — feasible in a few hours.

---

## Experimental Conditions (Baselines + Methods)

| Condition | Description | Based On |
|---|---|---|
| **B1: Zero-shot** | Single-pass generation, no refinement | Baseline |
| **B2: Chain-of-Thought** | Prompt with "think step by step" | Wang et al. 2023 |
| **SR-1** | Self-Refine with 1 iteration (generate → feedback → refine) | Madaan et al. 2023 |
| **SR-2** | Self-Refine with 2 iterations | Madaan et al. 2023 |
| **SR-3** | Self-Refine with 3 iterations | Madaan et al. 2023 |
| **RaR** | Rephrase-and-Respond (rephrase input, then answer) | Deng et al. 2023 |

This gives: 2 baselines + 4 refinement conditions × 2 models = **12 experiment cells**.

---

## Self-Refine Loop Implementation

```
Input: source text / question
│
├── Step 1: GENERATE — prompt model to produce output y₀
│
├── Step 2: FEEDBACK — prompt model to critique y₀
│          "What is wrong with this output? Be specific."
│
├── Step 3: REFINE — prompt model to fix y₀ using the feedback
│          → produces y₁
│
└── Repeat steps 2–3 for k iterations (k = 1, 2, 3)
```

Each intermediate output (y₀, y₁, y₂, y₃) is saved for analysis.

---

## Notebook Structure (`.ipynb`)

### Cell Groups

```
0. Setup & Config
   - Install: requests, rouge-score, bert-score, datasets, numpy, pandas, matplotlib
   - Set Ollama base URL (http://localhost:11434)
   - Set model name, dataset size, iterations

1. Ollama API Wrapper
   - call_ollama(prompt, model, temperature) → string
   - Simple HTTP POST to /api/generate

2. Dataset Loading
   - Load CNN/DailyMail subset (50 articles + reference summaries)
   - Load SQuAD subset (50 QA pairs)

3. Prompt Templates
   - zero_shot_prompt(article)
   - cot_prompt(article)
   - feedback_prompt(article, output)
   - refine_prompt(article, output, feedback)
   - rar_rephrase_prompt(question)

4. Experiment Runner
   - run_baseline(samples, model) → results dict
   - run_self_refine(samples, model, k_iterations) → results dict
     → saves all intermediate outputs
   - run_rar(samples, model) → results dict

5. Evaluation
   - compute_rouge(hypothesis, reference) → {rouge-1, rouge-l}
   - compute_bertscore(hypothesis, reference) → F1
   - compute_iteration_diversity(outputs_list) → avg embedding distance
     (catches reasoning path collapse from the literature gap)

6. Results Tables
   - Per-condition ROUGE/BERTScore means + std
   - Iteration-over-iteration score delta (does it improve or degrade?)
   - Diversity score per iteration (collapse detection)

7. Visualizations
   - Line plot: ROUGE vs. iteration number (for SR-1, SR-2, SR-3)
   - Bar chart: all conditions comparison
   - Heatmap: per-sample improvement/degradation

8. Error Analysis (MANDATORY for high grade)
   - Categorize failures: vague feedback / over-correction / semantic drift
   - Show 3–5 concrete examples: good refinement, bad refinement, collapse
   - Quote the feedback the model generated and show what changed

9. Ablation Study
   - Does feedback quality matter? Compare: specific feedback vs. generic feedback
   - Does temperature affect refinement? Compare temp=0.3 vs. temp=0.7

10. Discussion & Findings Summary
    - Answer the research question with evidence
    - Reference specific gaps from literature (why small models fail)
```

---

## Key Metrics to Track

| Metric | Purpose | How |
|---|---|---|
| ROUGE-1 / ROUGE-L | Overlap with reference summary | `rouge-score` library |
| BERTScore F1 | Semantic similarity to reference | `bert-score` library |
| Iteration Quality Delta | Does each iteration actually improve? | score(yᵢ) - score(yᵢ₋₁) |
| Diversity Score | Detect reasoning path collapse | cosine distance between consecutive output embeddings |
| Constraint Satisfaction | For constrained generation task | Rule-based regex check |
| Feedback Specificity | Is the feedback actionable? | Avg length + keyword check |

---

## Hypotheses (to test and confirm/reject in the paper)

**H1:** Small LLMs (3B) show minimal or no improvement from self-refinement, and may degrade by iteration 3.  
**H2:** Medium models (7B) show modest improvement in early iterations (k=1,2) but plateau or collapse at k=3.  
**H3:** RaR (input-side refinement) is more effective than output-side Self-Refine for small models.  
**H4:** Feedback quality degrades with model size — small models produce vague, repetitive feedback.

These hypotheses directly address the **Scale Gate** gap identified in the literature analysis across Self-Refine, Self-Polish, and all other papers.

---

## Timeline for Implementation

| Date | Milestone |
|---|---|
| April 19 (today) | `implementation.ipynb` submitted with all cells running |
| April 25 | Run full experiments, collect all results |
| April 28 | Error analysis + plots finalized |
| May 3 | IEEE paper in LaTeX submitted |

---

## Dependencies to Install

```bash
pip install requests rouge-score bert-score datasets numpy pandas matplotlib seaborn sentence-transformers
```

Ollama must be running on the target device:
```bash
ollama serve
ollama pull llama3.2:3b
ollama pull mistral:7b   # optional, for comparison
```

---

## File Structure

```
NLP Research Paper/
├── plan.md                    ← this file
├── implementation.ipynb       ← main notebook (to be built)
├── data/
│   ├── cnn_samples.json       ← 50-100 CNN/DM articles + references
│   └── squad_samples.json     ← 50 QA pairs
├── results/
│   ├── raw_outputs.json       ← all model outputs per condition
│   └── metrics.csv            ← computed scores
└── Combined_Analysis_All_10_Papers.docx
```

---

## Connection to Literature (for Paper writing later)

| Our experiment | Directly addresses gap from |
|---|---|
| Small model (3B/7B) self-refine test | Self-Refine Gap 6 (no open-weight model tests) |
| Iteration diversity tracking | Cross-paper gap: "Reasoning Path Collapse" (section 2.3) |
| Stopping at k=3 | Self-Refine Gap 1 (cost), Self-Polish Gap 2 (arbitrary T=2) |
| Feedback quality analysis | Self-Refine Gap 7 (hidden prompt engineering labor) |
| RaR vs. Self-Refine comparison | SSR / PASR / Reflexion unified gap: no comparison at small scale |
| Collapse detection metric | Survey Paper Gap 3 (model collapse underweighted) |
