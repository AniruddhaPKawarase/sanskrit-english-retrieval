"""Generate notebooks/sanskrit_retrieval_finetune_mark3.ipynb — the research-backed
"last try". Improvements over mark1/mark2:
  * Loss: CachedGISTEmbedLoss (guide=multilingual-e5-base) — masks near-duplicate
    in-batch false negatives (the correct fix the hard-neg mining could not achieve),
    wrapped in MatryoshkaLoss; effective batch 256 (mini 32); 2 epochs.
  * Two-stage reranking (BAAI/bge-reranker-v2-m3) with a gated paired before/after.
  * Matryoshka truncation eval; e5-base + INT8 bonuses. Hard-neg cell removed.
Neutral, imperative voice (no first/second-person pronouns). Run: python scripts/build_notebook_mark3.py
"""
from __future__ import annotations

import json
import os

MD, CODE = "markdown", "code"


def cell(kind, src):
    src = src.strip("\n") + "\n"
    base = {"metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == CODE:
        base.update({"cell_type": "code", "outputs": [], "execution_count": None})
    else:
        base.update({"cell_type": "markdown"})
    return base


CELLS = [
    (MD, """
# Sanskrit / English Semantic Retrieval — mark3 (research-backed recipe)

Improvements over prior runs, each grounded in current best practice:
- **Loss = CachedGISTEmbedLoss** (frozen guide = `multilingual-e5-base`) wrapped in **MatryoshkaLoss**.
  GIST uses the guide model to mask near-duplicate in-batch false negatives — the correct fix for the
  epic corpus's duplicate density (naive/margin hard-negative mining could not achieve this). Cached =
  large effective batch on a T4; Matryoshka = truncatable embeddings.
- **Effective batch 256** (mini-batch 32) and **2 epochs** — more/harder in-batch negatives.
- **Two-stage reranking** (`BAAI/bge-reranker-v2-m3`, same XLM-R family as e5) with a **gated** paired
  before/after — reranking ships only if it improves cross-lingual ranking.

**Run order:** Setup → Data → Tokenizer → Baseline → Train (GIST) → Before/after → OOD → Rerank →
RAG demo → Failure analysis → Save → Bonus. Set the runtime to GPU (T4).
"""),

    (MD, """
## 1. Setup — mount Google Drive, set the project folder, install deps
Upload the `sanskrit-assignment` folder to Drive and set **`PROJECT_DIR`** below. All outputs persist
inside that folder on Drive across runtime resets.
"""),
    (CODE, """
from google.colab import drive
drive.mount('/content/drive')

import os, sys
# vvv EDIT to match the uploaded folder path (MyDrive = /content/drive/MyDrive/) vvv
PROJECT_DIR = "/content/drive/MyDrive/sanskrit-assignment"
assert os.path.isdir(PROJECT_DIR), f"'{PROJECT_DIR}' not found — upload the folder and fix PROJECT_DIR."
os.chdir(PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

!pip -q install "sentence-transformers>=3.0" "datasets>=2.20" faiss-cpu indic-transliteration sacrebleu

import torch
print("Project:", PROJECT_DIR)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
"""),

    (MD, """
### Hugging Face login (optional — unlocks the gated OOD eval sets)
Enables IN22 / FLORES for the out-of-domain number. Optional: without it the OOD cell falls back to the
ungated Bhagavad Gita cross-domain eval. Prereqs (once): a **Read** token at hf.co/settings/tokens and
"Agree and access" on the IN22 and FLORES dataset pages.
"""),
    (CODE, """
from huggingface_hub import notebook_login
notebook_login()   # paste a READ token; or: from huggingface_hub import login; login(token="hf_xxx")
"""),
    (CODE, """
from sanskrit_retrieval.config import DEFAULT
from sanskrit_retrieval import data, model as M, train as T, evaluate as E, index as IX, rag, rerank
import numpy as np, random, json, pandas as pd

# mark3 recipe overrides (research-backed). loss_type is passed to T.train below.
cfg = DEFAULT.with_(
    output_dir=os.path.join(PROJECT_DIR, "artifacts"),
    run_name="e5-small-mark3-gist",
    batch_size=256,        # EFFECTIVE batch — Cached loss keeps VRAM at mini_batch_size
    mini_batch_size=32,    # drop to 16 if the T4 OOMs
    epochs=2,
    matryoshka=True,       # truncatable embeddings + mild regularization
    guide_model="intfloat/multilingual-e5-base",  # frozen GIST guide (Indic-aware)
)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(cfg.output_dir, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
print("checkpoints ->", cfg.output_dir, "| results ->", RESULTS_DIR)
cfg
"""),

    (MD, "## 2. Data (Itihasa ~93K aligned pairs → NFC → directional prefixed pairs → 25% IAST → dedup)"),
    (CODE, """
sa_train, en_train = data.load_itihasa("train")
sa_test,  en_test  = data.load_itihasa("test")
pairs = data.dedup_pairs(data.build_pairs(sa_train, en_train, cfg) + data.augment_iast(sa_train, en_train, cfg))
print(f"train rows {len(sa_train):,} | test rows {len(sa_test):,} | training pairs {len(pairs):,}")
"""),

    (MD, "## 3. Tokenizer diagnostic (Devanagari fertility)"),
    (CODE, """
base_model = M.load_model(cfg)
fert = M.devanagari_fertility(base_model, " ".join(sa_train[:50]).split())
print(f"e5-small Devanagari fertility: {fert:.2f} subword-tokens/word")
"""),

    (MD, "## 4. Baseline evaluation (in-domain Itihasa test, before fine-tuning)"),
    (CODE, """
ir_indomain = E.build_ir_eval(sa_test, en_test, cfg)
baseline = E.evaluate_model(base_model, ir_indomain)
{k: round(v, 4) for k, v in baseline.items() if "cos" in k}
"""),

    (MD, """
## 5. Train — CachedGISTEmbedLoss + Matryoshka (mark3 recipe)
The frozen guide model masks near-duplicate in-batch negatives (false-negative fix); Cached gives a
large effective batch on the T4; Matryoshka makes embeddings truncatable. Loads a guide model, so
expect extra VRAM — reduce `mini_batch_size` to 16 if the run OOMs.
"""),
    (CODE, """
train_model = M.load_model(cfg)
out_path = T.train(train_model, pairs, cfg, loss_type="cached_gist")
print("saved fine-tuned model to:", out_path)
tuned_model = M.load_model(cfg, checkpoint=out_path)
"""),

    (MD, "## 6. Before/after — the headline result (in-domain)"),
    (CODE, """
ba = E.before_after(base_model, tuned_model, ir_indomain)
df = pd.DataFrame([{"metric": k, "before": ba["before"][k], "after": ba["after"][k], "delta": ba["delta"][k]}
                   for k in ba["before"] if "cos" in k]).round(4)
df.to_csv(os.path.join(RESULTS_DIR, "metrics_indomain.csv"), index=False)
json.dump(ba, open(os.path.join(RESULTS_DIR, "before_after_indomain.json"), "w"), indent=2)
df
"""),
    (CODE, """
import matplotlib.pyplot as plt
plot_df = df[df["metric"].str.contains("recall|ndcg|mrr", case=False)].set_index("metric")[["before","after"]]
ax = plot_df.plot(kind="bar", figsize=(11,4), title="mark3 — Sanskrit->English retrieval: base vs fine-tuned (in-domain)")
ax.set_ylabel("score"); plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "before_after_indomain.png"), dpi=150, bbox_inches="tight")
plt.show()
"""),

    (MD, "**Out-of-domain — the trustworthy generalization number.** IN22 → FLORES (gated) → ungated **Bhagavad Gita** cross-domain → Itihasa-slice proxy. The eval set used is printed and saved."),
    (CODE, """
try:
    sa_o, en_o = data.load_in22();               eval_tag = "AI4Bharat IN22 (out-of-domain)"
except Exception:
    try:
        sa_o, en_o = data.load_flores_sanskrit("devtest"); eval_tag = "FLORES san_Deva (out-of-domain)"
    except Exception:
        try:
            sa_o, en_o = data.load_gita();       eval_tag = "Bhagavad Gita (ungated cross-domain)"
        except Exception:
            half = len(sa_test) // 2
            sa_o, en_o = sa_test[half:], en_test[half:]; eval_tag = "Itihasa held-out slice (OOD PROXY)"
ir_ood = E.build_ir_eval(sa_o, en_o, cfg)
ood = E.before_after(base_model, tuned_model, ir_ood)
print("Eval set:", eval_tag)
ood_df = pd.DataFrame([{"metric": k, "before": ood["before"][k], "after": ood["after"][k]}
                       for k in ood["before"] if "cos" in k]).round(4)
ood_df.to_csv(os.path.join(RESULTS_DIR, "metrics_ood.csv"), index=False)
json.dump({"eval_set": eval_tag, **ood}, open(os.path.join(RESULTS_DIR, "before_after_ood.json"), "w"), indent=2)
ood_df
"""),

    (MD, """
## 7. Two-stage reranking (gated)
Retrieve top-N with the fine-tuned bi-encoder, then rerank with `bge-reranker-v2-m3` on the SAME
candidate set (paired before/after on Sanskrit→English in-domain). A cross-encoder that has not seen
Sanskrit can REGRESS cross-lingual ranking — **ship the reranker only if the delta is positive.**
"""),
    (CODE, """
reranker = rerank.load_reranker(cfg)
rr = rerank.evaluate_rerank(tuned_model, reranker, sa_test, en_test, cfg)
rr_df = pd.DataFrame([{"metric": k, "bi-encoder": rr["before"][k], "+ reranker": rr["after"][k],
                       "delta": round(rr["after"][k] - rr["before"][k], 4)} for k in rr["before"]])
rr_df.to_csv(os.path.join(RESULTS_DIR, "metrics_rerank.csv"), index=False)
json.dump(rr, open(os.path.join(RESULTS_DIR, "rerank.json"), "w"), indent=2)
print(f"reranked top-{rr['top_n']} on {rr['n_queries']} queries  |  SHIP only if delta > 0")
rr_df
"""),

    (MD, """
## 8. Mini-RAG demo (Bhagavad Gita corpus)
Index Gita English translations; retrieve for natural-language queries incl. the spec's karma example.
Optional reranking of the top-k shown inline.
"""),
    (CODE, """
try:
    gita_sa, gita_en = data.load_gita()
    passages = gita_en; sanskrit_query = gita_sa[0]; demo_tag = f"Bhagavad Gita ({len(passages)} verses)"
except Exception as e:
    print("Gita load failed -> Itihasa slice. Detail:", e)
    passages = en_test[: cfg.eval_max_corpus]; sanskrit_query = sa_test[0]; demo_tag = "Itihasa slice (fallback)"
print("Demo corpus:", demo_tag)

vindex = IX.VerseIndex(tuned_model, passages, cfg)
demo = {"corpus": demo_tag, "queries": []}
for q in ["what does this verse say about karma?", "the duty of a warrior in battle", sanskrit_query]:
    hits = vindex.search(q, k=cfg.rerank_top_n)
    reranked = rerank.rerank(reranker, q, [h["text"] for h in hits])[:3]   # rerank the shortlist
    demo["queries"].append({"query": q, "reranked_top3": reranked})
    print("Q:", q[:70])
    for h in reranked:
        print(f"   [{h['rank']}] {h['score']:.3f}  {h['text'][:80]}")
    print()
json.dump(demo, open(os.path.join(RESULTS_DIR, "rag_demo_samples.json"), "w"), ensure_ascii=False, indent=2)
"""),

    (MD, """
## 9. Failure analysis — transliteration robustness
Query the same in-corpus verse in Devanagari and in IAST; both should retrieve the same top-1 if the
model is script-robust (25% IAST augmentation).
"""),
    (CODE, """
from sanskrit_retrieval.normalize import to_iast, detect_script
dv = sanskrit_query; iast = to_iast(dv)
dv_top = vindex.search(dv, k=1)[0]["text"]; iast_top = vindex.search(iast, k=1)[0]["text"]
print("scripts:", detect_script(dv), "/", detect_script(iast))
print("Devanagari top-1:", dv_top[:80])
print("IAST       top-1:", iast_top[:80])
print("same verse retrieved by both scripts:", dv_top == iast_top)
"""),

    (MD, """
## 10. Save & next
Persisted to Drive `results/`: metrics_indomain.csv, metrics_ood.csv, metrics_rerank.csv,
before_after_*.json, rerank.json, before_after_indomain.png, rag_demo_samples.json, and the
bonus_*.csv below. Checkpoints in `artifacts/`. Log config + metrics in `report/REPORT.md`.

**Honesty note:** 1:1 verse retrieval is *translation retrieval* — easier than open thematic QA.
Absolute scores stay modest; the result is the *relative* before/after and the methodology.
"""),

    (MD, """
## 11. Bonus experiments
### 11a. Matryoshka truncation — recall at reduced dims (storage/speed lever)
Because training wrapped the loss in MatryoshkaLoss, embeddings truncate gracefully. Lower dims mean a
smaller/faster index at a measured recall cost.
"""),
    (CODE, """
from sanskrit_retrieval.model import encode_queries, encode_passages
from sanskrit_retrieval.evaluate import recall_at_k
corpus = en_test[: cfg.eval_max_corpus]; q = sa_test[: cfg.eval_max_corpus]; gold = list(range(len(corpus)))
P = encode_passages(tuned_model, corpus, cfg); Q = encode_queries(tuned_model, q, cfg)
def rank(Qe, Pe, k=10):
    s = Qe @ Pe.T; return [list(np.argsort(-r)[:k]) for r in s]
rows = []
for d in cfg.matryoshka_dims:
    Pd = P[:, :d] / np.linalg.norm(P[:, :d], axis=1, keepdims=True)
    Qd = Q[:, :d] / np.linalg.norm(Q[:, :d], axis=1, keepdims=True)
    rows.append({"dim": d, "recall@10": round(recall_at_k(rank(Qd, Pd), gold, 10), 4)})
mdf = pd.DataFrame(rows); mdf.to_csv(os.path.join(RESULTS_DIR, "bonus_matryoshka.csv"), index=False); mdf
"""),

    (MD, """
### 11b. e5-base with the mark3 recipe
Larger XLM-R backbone (~2.4× params), same GIST+Matryoshka recipe. Guide set to e5-small to save VRAM.
"""),
    (CODE, """
cfg_base = cfg.with_(base_model="intfloat/multilingual-e5-base", run_name="e5-base-mark3-gist",
                     guide_model="intfloat/multilingual-e5-small", mini_batch_size=16)
m_base = M.load_model(cfg_base)
out_base = T.train(m_base, pairs, cfg_base, loss_type="cached_gist")
tuned_base = M.load_model(cfg_base, checkpoint=out_base)
ev_small = E.evaluate_model(tuned_model, ir_indomain); ev_base = E.evaluate_model(tuned_base, ir_indomain)
cmp_df = pd.DataFrame([{"metric": k, "e5-small (tuned)": ev_small[k], "e5-base (tuned)": ev_base[k]}
              for k in ev_small if "cos" in k]).round(4)
cmp_df.to_csv(os.path.join(RESULTS_DIR, "bonus_e5base_vs_small.csv"), index=False); cmp_df
"""),

    (MD, """
### 11c. INT8 embedding quantization — recall retention + index shrink
"""),
    (CODE, """
try:
    from sentence_transformers.util.quantization import quantize_embeddings
except Exception:
    from sentence_transformers.quantization import quantize_embeddings
r_float = recall_at_k(rank(Q, P), gold, 10)
Pi = quantize_embeddings(P, precision="int8"); Qi = quantize_embeddings(Q, precision="int8")
r_int8 = recall_at_k(rank(Qi.astype(np.float32), Pi.astype(np.float32)), gold, 10)
quant = {"recall@10_float32": round(r_float, 4), "recall@10_int8": round(r_int8, 4),
         "retention": round(r_int8 / max(r_float, 1e-9), 4),
         "index_mb_float32": round(P.nbytes / 1e6, 2), "index_mb_int8": round(Pi.nbytes / 1e6, 2)}
json.dump(quant, open(os.path.join(RESULTS_DIR, "bonus_int8_quant.json"), "w"), indent=2); print(quant)
"""),

    (MD, """
### Note — why hard-negative mining was dropped
mark1/mark2 showed explicit hard-negative mining fails on this corpus: without a margin the mined
negatives are near-duplicate shlokas (false negatives, degrading training); with a strict margin
almost no valid negatives remain. **CachedGISTEmbedLoss (§5) solves the same problem correctly** — a
frozen guide model masks the false negatives inside the in-batch set, so no separate mining is needed.
"""),
]


def main():
    nb = {"cells": [cell(k, s) for k, s in CELLS],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "sanskrit_retrieval_finetune_mark3.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", os.path.normpath(out), f"({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
