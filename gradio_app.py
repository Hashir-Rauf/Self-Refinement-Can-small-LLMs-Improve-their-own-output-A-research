"""
Self-Refinement Gradio UI
Run: python3 gradio_app.py
Then open http://localhost:7860
"""

import os, json, time, threading
import numpy as np
import pandas as pd
import requests
import gradio as gr
from rouge_score import rouge_scorer as rs_module

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
DEFAULT_MODEL     = "llama3.2:3b"
TEMPERATURE       = 0.7
FEEDBACK_TEMP     = 0.3
MAX_TOKENS_GEN    = 256
MAX_TOKENS_FB     = 150
OLLAMA_TIMEOUT    = 180
SEED              = 42
RESULTS_DIR       = "results"
DATA_DIR          = "data"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

_rouge = rs_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

EXAMPLE_ARTICLE = (
    "Scientists at NASA announced Thursday the discovery of water ice deposits "
    "near the Moon's south pole, confirmed by the Lunar Reconnaissance Orbiter. "
    "The deposits, estimated at 600 million metric tons, are located in permanently "
    "shadowed craters where temperatures drop to -173°C. NASA administrator Jim "
    "Bridenstine called it a 'game changer' for future lunar missions, as the ice "
    "could be used for drinking water and converted into rocket fuel. The Artemis "
    "program, which aims to land astronauts on the Moon by 2026, will specifically "
    "target the south polar region. China and the European Space Agency also confirmed "
    "they are adjusting their lunar programs to focus on the same deposits."
)

EXAMPLE_REFERENCE = (
    "NASA confirmed water ice near the Moon's south pole using the Lunar Reconnaissance "
    "Orbiter. The 600 million metric ton deposit could support future lunar missions "
    "including the Artemis program targeting a 2026 Moon landing."
)

# ─── OLLAMA HELPERS ───────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str, temperature: float = TEMPERATURE,
                max_tokens: int = MAX_TOKENS_GEN) -> str:
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": SEED},
    }
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[ERROR: Ollama not running — start with `ollama serve`]"
    except requests.exceptions.Timeout:
        return "[ERROR: Request timed out]"
    except Exception as e:
        return f"[ERROR: {e}]"


def get_available_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        return names if names else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def check_ollama_status() -> str:
    models = get_available_models()
    if models[0].startswith("ERROR") or models == [DEFAULT_MODEL]:
        try:
            requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            return f"✅ Connected  |  Models: {', '.join(models)}"
        except Exception:
            return "❌ Ollama not reachable — run `ollama serve` on this device"
    return f"✅ Connected  |  Models: {', '.join(models)}"


# ─── ROUGE ────────────────────────────────────────────────────────────────────

def rouge(hyp: str, ref: str) -> dict:
    if not hyp.strip() or not ref.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    s = _rouge.score(ref, hyp)
    return {k: round(s[k].fmeasure, 4) for k in ("rouge1", "rouge2", "rougeL")}


def rouge_badge(scores: dict) -> str:
    return (f"ROUGE-1: **{scores['rouge1']:.4f}**  |  "
            f"ROUGE-2: **{scores['rouge2']:.4f}**  |  "
            f"ROUGE-L: **{scores['rougeL']:.4f}**")


# ─── PROMPT TEMPLATES ─────────────────────────────────────────────────────────

def P_zero_shot(article):
    return f"Summarize the following article in 2-3 sentences. Be concise and factual.\n\nArticle:\n{article}\n\nSummary:"

def P_cot(article):
    return (f"Read the article. First list the 2-3 most important facts, "
            f"then write a concise 2-3 sentence summary.\n\nArticle:\n{article}\n\nStep 1 – Key facts:\n1.")

def P_rar(article):
    return (f"Rephrase the task to be more specific, then complete it.\n\n"
            f"Task: Summarize this article.\n\nArticle:\n{article}\n\n"
            f"Rephrased task: Write a 2-3 sentence summary capturing the main event, "
            f"key people, and outcome.\n\nSummary:")

def P_feedback(article, output):
    return (f"You are a strict writing evaluator. List exactly 2 specific, actionable problems "
            f"with this summary.\nEach problem must name what is wrong "
            f"(e.g., 'Missing the name of X', 'Incorrect date').\n\n"
            f"Article:\n{article}\n\nSummary:\n{output}\n\nProblems:\n1.")

def P_refine(article, output, feedback):
    return (f"Rewrite the summary to fix the problems listed in the feedback. "
            f"Keep it to 2-3 sentences.\n\n"
            f"Article:\n{article}\n\nOriginal summary:\n{output}\n\n"
            f"Feedback:\n{feedback}\n\nImproved summary:")


# ─── PLAYGROUND LOGIC ─────────────────────────────────────────────────────────

def run_playground(article: str, reference: str, condition: str, model: str,
                   k_iterations: int):
    """
    Generator: yields (output_md, trace_md, scores_md) as each step completes.
    Used with gr.Interface streaming.
    """
    if not article.strip():
        yield "⚠️ Please enter an article.", "", ""
        return
    if "ERROR" in model or not model:
        yield "⚠️ No model selected or Ollama not connected.", "", ""
        return

    trace_lines = []
    outputs     = []

    def emit(out_md, trace_md, scores_md):
        return out_md, trace_md, scores_md

    # ── Initial generation ──
    yield "⏳ Generating initial output (y₀)…", "", ""
    t0 = time.time()
    if condition == "Zero-shot":
        prompt = P_zero_shot(article)
    elif condition == "Chain-of-Thought":
        prompt = P_cot(article)
    else:  # RaR
        prompt = P_rar(article)

    y0 = call_ollama(prompt, model=model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS_GEN)
    outputs.append(y0)
    elapsed = round(time.time() - t0, 1)
    trace_lines.append(f"**y₀ (initial, {elapsed}s)**\n\n{y0}\n\n---")

    r0 = rouge(y0, reference) if reference.strip() else {}
    scores_md = rouge_badge(r0) if r0 else "*Add a reference summary to see ROUGE scores.*"

    if condition in ("Zero-shot", "Chain-of-Thought", "Rephrase-and-Respond"):
        final_md = f"### Final Output\n\n{y0}"
        yield final_md, "\n\n".join(trace_lines), scores_md
        return

    # ── Self-Refine iterations ──
    for i in range(k_iterations):
        yield (f"⏳ Iteration {i+1}/{k_iterations} — generating feedback…",
               "\n\n".join(trace_lines), scores_md)

        t0 = time.time()
        fb = call_ollama(P_feedback(article, outputs[-1]),
                         model=model, temperature=FEEDBACK_TEMP, max_tokens=MAX_TOKENS_FB)
        fb_elapsed = round(time.time() - t0, 1)
        trace_lines.append(f"**Feedback fb{i+1} ({fb_elapsed}s)**\n\n{fb}\n\n---")

        yield (f"⏳ Iteration {i+1}/{k_iterations} — refining…",
               "\n\n".join(trace_lines), scores_md)

        t0 = time.time()
        refined = call_ollama(P_refine(article, outputs[-1], fb),
                              model=model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS_GEN)
        ref_elapsed = round(time.time() - t0, 1)
        outputs.append(refined)

        ri = rouge(refined, reference) if reference.strip() else {}
        delta_str = ""
        if r0 and ri:
            d = ri["rouge1"] - r0["rouge1"]
            delta_str = f"  (Δ ROUGE-1 from y₀: **{d:+.4f}**)"
        trace_lines.append(f"**y{i+1} (refined, {ref_elapsed}s)**{delta_str}\n\n{refined}\n\n---")
        scores_md = rouge_badge(ri) + delta_str if ri else scores_md

        yield ("\n\n".join(trace_lines),
               "\n\n".join(trace_lines),
               scores_md)

    final_out = outputs[-1]
    final_md = (
        f"### Final Output (y{k_iterations})\n\n{final_out}\n\n"
        f"---\n**Initial output (y₀)**\n\n{outputs[0]}"
    )
    rn = rouge(final_out, reference) if reference.strip() else {}
    if r0 and rn:
        d = rn["rouge1"] - r0["rouge1"]
        scores_md = (
            f"**y₀:** {rouge_badge(r0)}\n\n"
            f"**y{k_iterations}:** {rouge_badge(rn)}\n\n"
            f"**Net ROUGE-1 change:** {'▲' if d > 0 else '▼'} {d:+.4f}"
        )
    yield final_md, "\n\n".join(trace_lines), scores_md


# ─── BATCH RUNNER LOGIC ───────────────────────────────────────────────────────

def load_sample_articles(n: int = 5) -> list:
    """Load cached CNN samples; fall back to built-in example."""
    path = os.path.join(DATA_DIR, "cnn_samples.json")
    if os.path.exists(path):
        with open(path) as f:
            samples = json.load(f)
        return samples[:n]
    return [{"id": "example_0", "article": EXAMPLE_ARTICLE, "reference": EXAMPLE_REFERENCE}]


def run_batch(model: str, conditions_str: str, n_samples: int, progress=gr.Progress()):
    """Run batch experiments and return a results DataFrame."""
    conditions_map = {
        "Zero-shot": "zero_shot",
        "CoT": "cot",
        "RaR": "rar",
        "SR k=1": "sr_1",
        "SR k=2": "sr_2",
        "SR k=3": "sr_3",
    }
    selected = [c.strip() for c in conditions_str.split(",") if c.strip() in conditions_map]
    if not selected:
        return pd.DataFrame({"Error": ["No valid conditions selected."]}), "No conditions selected."

    samples = load_sample_articles(int(n_samples))
    rows    = []
    total   = len(selected) * len(samples)
    done    = 0

    for cond_label in selected:
        cond_key = conditions_map[cond_label]
        for sample in samples:
            progress(done / total, desc=f"{cond_label} — sample {done % len(samples) + 1}/{len(samples)}")
            article   = sample["article"]
            reference = sample.get("reference", "")

            if cond_key == "zero_shot":
                output = call_ollama(P_zero_shot(article), model=model)
            elif cond_key == "cot":
                output = call_ollama(P_cot(article), model=model)
            elif cond_key == "rar":
                output = call_ollama(P_rar(article), model=model)
            else:
                k = int(cond_key.split("_")[1])
                outputs = [call_ollama(P_zero_shot(article), model=model)]
                for _ in range(k):
                    fb = call_ollama(P_feedback(article, outputs[-1]),
                                     model=model, temperature=FEEDBACK_TEMP,
                                     max_tokens=MAX_TOKENS_FB)
                    refined = call_ollama(P_refine(article, outputs[-1], fb), model=model)
                    outputs.append(refined)
                output = outputs[-1]

            r = rouge(output, reference)
            rows.append({
                "id":        sample["id"],
                "condition": cond_label,
                "ROUGE-1":   r["rouge1"],
                "ROUGE-2":   r["rouge2"],
                "ROUGE-L":   r["rougeL"],
                "output":    output[:120] + "…",
            })
            done += 1

    df = pd.DataFrame(rows)
    summary = df.groupby("condition")[["ROUGE-1", "ROUGE-2", "ROUGE-L"]].mean().round(4)

    # Save
    df.to_csv(f"{RESULTS_DIR}/batch_results.csv", index=False)
    summary_str = summary.to_string()
    return df, summary_str


# ─── GRADIO UI ────────────────────────────────────────────────────────────────

def build_ui():
    models = get_available_models()

    with gr.Blocks(
        title="Self-Refinement: Can Small LLMs Improve Their Own Output?",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .header-box { background: linear-gradient(90deg,#1a3a5c,#2e6da4);
                      padding:20px 24px; border-radius:10px; margin-bottom:16px; }
        .header-box h1 { color:#fff; margin:0; font-size:1.4em; }
        .header-box p  { color:#c8dff5; margin:6px 0 0; font-size:0.9em; }
        .metric-box    { background:#f0f7ff; border:1px solid #b8d4f0;
                         border-radius:8px; padding:12px 16px; }
        footer { display:none !important; }
        """
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div class="header-box">
          <h1>🔬 Self-Refinement: Can Small LLMs Improve Their Own Output?</h1>
          <p>CS4063 NLP Project &nbsp;·&nbsp; Hashir Rauf · Bilal Ahmad · Muhammad Saad &nbsp;|&nbsp; BDS-6B</p>
        </div>
        """)

        # ── Status bar ──
        with gr.Row():
            status_box = gr.Markdown(value=check_ollama_status(), elem_classes=["metric-box"])
            refresh_btn = gr.Button("🔄 Refresh Status", scale=0, size="sm")
        refresh_btn.click(fn=check_ollama_status, outputs=status_box)

        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tabs():

            # ── Tab 1: Playground ──────────────────────────────────────────────
            with gr.Tab("🧪 Playground — Single Article"):
                gr.Markdown(
                    "Run any condition on a single article and **watch each iteration happen live.**\n\n"
                    "The **Trace** column shows every intermediate output and feedback so you can "
                    "see exactly what the model is doing at each step."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        article_in = gr.Textbox(
                            label="📰 Article",
                            placeholder="Paste a news article here…",
                            lines=8,
                            value=EXAMPLE_ARTICLE,
                        )
                        reference_in = gr.Textbox(
                            label="📌 Reference Summary (optional — needed for ROUGE scores)",
                            placeholder="Gold-standard summary for evaluation…",
                            lines=3,
                            value=EXAMPLE_REFERENCE,
                        )
                        with gr.Row():
                            model_dd = gr.Dropdown(
                                choices=models,
                                value=models[0],
                                label="🤖 Model",
                            )
                            cond_dd = gr.Dropdown(
                                choices=["Zero-shot", "Chain-of-Thought",
                                         "Rephrase-and-Respond",
                                         "Self-Refine k=1", "Self-Refine k=2",
                                         "Self-Refine k=3"],
                                value="Self-Refine k=3",
                                label="⚙️ Condition",
                            )
                            k_slider = gr.Slider(
                                minimum=1, maximum=3, step=1, value=3,
                                label="Iterations (for Self-Refine)",
                                visible=True,
                            )
                        run_btn = gr.Button("▶  Run", variant="primary", size="lg")

                    with gr.Column(scale=4):
                        with gr.Tabs():
                            with gr.Tab("📄 Final Output"):
                                output_box = gr.Markdown(
                                    value="*Output will appear here.*",
                                    label="Output",
                                )
                            with gr.Tab("🔁 Iteration Trace"):
                                trace_box = gr.Markdown(
                                    value="*Step-by-step trace will appear here.*",
                                )
                        scores_box = gr.Markdown(
                            value="*Scores will appear here after running.*",
                            elem_classes=["metric-box"],
                        )

                # Show/hide k_slider based on condition
                def toggle_slider(cond):
                    return gr.update(visible="Self-Refine" in cond)
                cond_dd.change(toggle_slider, inputs=cond_dd, outputs=k_slider)

                def run_wrapper(article, reference, condition, model, k):
                    cond_key = condition.replace("Self-Refine k=", "Self-Refine").replace(" ", "_").lower()
                    # Normalize condition names
                    cond_map = {
                        "zero-shot": "Zero-shot",
                        "chain-of-thought": "Chain-of-Thought",
                        "rephrase-and-respond": "Rephrase-and-Respond",
                        "self-refine_k=1": "Self-Refine",
                        "self-refine_k=2": "Self-Refine",
                        "self-refine_k=3": "Self-Refine",
                    }
                    # Map UI label to internal condition
                    ui_to_internal = {
                        "Zero-shot":            "Zero-shot",
                        "Chain-of-Thought":     "Chain-of-Thought",
                        "Rephrase-and-Respond": "Rephrase-and-Respond",
                        "Self-Refine k=1":      "Self-Refine",
                        "Self-Refine k=2":      "Self-Refine",
                        "Self-Refine k=3":      "Self-Refine",
                    }
                    internal_cond = ui_to_internal.get(condition, condition)
                    k_val = int(condition[-1]) if "Self-Refine k=" in condition else int(k)
                    yield from run_playground(article, reference, internal_cond, model, k_val)

                run_btn.click(
                    fn=run_wrapper,
                    inputs=[article_in, reference_in, cond_dd, model_dd, k_slider],
                    outputs=[output_box, trace_box, scores_box],
                )

            # ── Tab 2: Batch Runner ────────────────────────────────────────────
            with gr.Tab("🗂 Batch Experiment Runner"):
                gr.Markdown(
                    "Run multiple conditions on CNN/DailyMail samples (loaded from `data/cnn_samples.json`).\n\n"
                    "> **Note:** Run the notebook first to download and cache the CNN dataset (`data/cnn_samples.json`). "
                    "If the file is missing, the built-in example article is used instead."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        batch_model_dd = gr.Dropdown(
                            choices=models, value=models[0], label="🤖 Model"
                        )
                        batch_conds = gr.CheckboxGroup(
                            choices=["Zero-shot", "CoT", "RaR", "SR k=1", "SR k=2", "SR k=3"],
                            value=["Zero-shot", "CoT", "SR k=1", "SR k=3"],
                            label="Conditions to run",
                        )
                        batch_n = gr.Slider(
                            minimum=1, maximum=50, step=1, value=5,
                            label="Number of samples",
                            info="Keep ≤10 for a quick test; 50 for full experiment",
                        )
                        batch_btn = gr.Button("▶  Run Batch", variant="primary")

                    with gr.Column(scale=3):
                        batch_table  = gr.Dataframe(label="Per-sample results", wrap=True)
                        batch_summary = gr.Textbox(
                            label="Summary (mean ROUGE per condition)",
                            lines=10, interactive=False,
                        )

                def batch_wrapper(model, conds, n):
                    conds_str = ", ".join(conds)
                    df, summary = run_batch(model, conds_str, n)
                    return df, summary

                batch_btn.click(
                    fn=batch_wrapper,
                    inputs=[batch_model_dd, batch_conds, batch_n],
                    outputs=[batch_table, batch_summary],
                )

            # ── Tab 3: Compare Outputs ─────────────────────────────────────────
            with gr.Tab("⚖️ Side-by-Side Compare"):
                gr.Markdown(
                    "Run **two conditions in parallel** on the same article to directly compare outputs."
                )
                with gr.Row():
                    cmp_article = gr.Textbox(
                        label="Article", lines=6, value=EXAMPLE_ARTICLE, scale=2
                    )
                    cmp_ref = gr.Textbox(
                        label="Reference (optional)", lines=3,
                        value=EXAMPLE_REFERENCE, scale=1
                    )
                with gr.Row():
                    cmp_model = gr.Dropdown(choices=models, value=models[0], label="Model")
                    cmp_cond_a = gr.Dropdown(
                        choices=["Zero-shot", "Chain-of-Thought", "Rephrase-and-Respond",
                                 "Self-Refine k=1", "Self-Refine k=2", "Self-Refine k=3"],
                        value="Zero-shot", label="Condition A"
                    )
                    cmp_cond_b = gr.Dropdown(
                        choices=["Zero-shot", "Chain-of-Thought", "Rephrase-and-Respond",
                                 "Self-Refine k=1", "Self-Refine k=2", "Self-Refine k=3"],
                        value="Self-Refine k=3", label="Condition B"
                    )
                    cmp_btn = gr.Button("▶  Compare", variant="primary")

                with gr.Row():
                    cmp_out_a = gr.Textbox(label="Output A", lines=8, interactive=False)
                    cmp_out_b = gr.Textbox(label="Output B", lines=8, interactive=False)
                with gr.Row():
                    cmp_score_a = gr.Markdown("*—*", elem_classes=["metric-box"])
                    cmp_score_b = gr.Markdown("*—*", elem_classes=["metric-box"])

                def run_compare(article, reference, model, cond_a, cond_b):
                    def single(cond):
                        cond_map = {
                            "Zero-shot": "Zero-shot",
                            "Chain-of-Thought": "Chain-of-Thought",
                            "Rephrase-and-Respond": "Rephrase-and-Respond",
                        }
                        if cond in cond_map:
                            internal = cond_map[cond]
                            k = 0
                        else:
                            internal = "Self-Refine"
                            k = int(cond[-1])
                        results = list(run_playground(article, reference, internal, model, max(k, 1)))
                        if results:
                            out_md, _, score_md = results[-1]
                            # Extract plain text from markdown
                            out_text = out_md.replace("### Final Output\n\n", "")
                            out_text = out_text.split("\n\n---")[0]
                            return out_text, score_md
                        return "No output.", ""

                    out_a, sc_a = single(cond_a)
                    out_b, sc_b = single(cond_b)
                    return out_a, out_b, sc_a, sc_b

                cmp_btn.click(
                    fn=run_compare,
                    inputs=[cmp_article, cmp_ref, cmp_model, cmp_cond_a, cmp_cond_b],
                    outputs=[cmp_out_a, cmp_out_b, cmp_score_a, cmp_score_b],
                )

            # ── Tab 4: Results Viewer ──────────────────────────────────────────
            with gr.Tab("📊 Results Viewer"):
                gr.Markdown("View and reload saved batch results from `results/batch_results.csv`.")
                reload_btn = gr.Button("🔄 Reload Results", size="sm")
                results_table = gr.Dataframe(label="Saved Results")
                results_summary = gr.Textbox(label="Mean ROUGE per condition", lines=10,
                                             interactive=False)

                def load_results():
                    path = f"{RESULTS_DIR}/batch_results.csv"
                    if not os.path.exists(path):
                        return pd.DataFrame({"info": ["No results yet — run Batch Experiment first."]}), ""
                    df = pd.read_csv(path)
                    numeric_cols = [c for c in ["ROUGE-1", "ROUGE-2", "ROUGE-L"] if c in df.columns]
                    if "condition" in df.columns and numeric_cols:
                        summary = df.groupby("condition")[numeric_cols].mean().round(4).to_string()
                    else:
                        summary = "Cannot summarize — missing expected columns."
                    return df, summary

                reload_btn.click(fn=load_results, outputs=[results_table, results_summary])
                demo.load(fn=load_results, outputs=[results_table, results_summary])

        # ── Footer ──
        gr.Markdown(
            "---\n"
            "**Research Question:** Can small LLMs improve their own output through iterative self-refinement?  \n"
            "**Literature Gap:** Self-Refine / Self-Polish / Reflexion were only tested on GPT-3.5/GPT-4. "
            "This project tests at 3B–7B local scale via Ollama.",
            elem_classes=["metric-box"],
        )

    return demo


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Gradio UI...")
    print(f"Ollama status: {check_ollama_status()}")
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
