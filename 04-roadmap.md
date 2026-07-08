# Phase 4 — Execution Roadmap (Option 2)

This document sequences the build into checkpoint-gated categories, defines the iteration loop, maps deliverables 1:1 to the assignment's Submission Requirements, and reviews the plan against the 12 production perspectives.

**Grounding:** [`03-technical-design.md`](03-technical-design.md) · [`05-production-system-design.md`](05-production-system-design.md).
**Repo root:** `sanskrit-assignment/` (code + docs ship as one coherent bundle).

---

## Repo structure (target)

```
docs/sanskrit-assignment/
├── 01..05 *.md                     # phase docs (research → decision → design → roadmap → prod design)
├── README.md                       # setup + how to run
├── report/REPORT.md                # the graded short report (9 sections from spec)
├── notebooks/
│   └── sanskrit_retrieval_finetune.ipynb   # the Colab: end-to-end, richly explained
├── src/sanskrit_retrieval/
│   ├── config.py                   # typed config, no hardcoded values
│   ├── data.py                     # load Itihasa/FLORES/Gita, normalize, dedup, pairs
│   ├── normalize.py                # Unicode NFC + Devanagari/IAST transliteration
│   ├── model.py                    # load e5, prefix + L2-norm wrappers
│   ├── train.py                    # MNRL contrastive fine-tune
│   ├── evaluate.py                 # IR metrics: Recall@K / MRR / nDCG, before/after
│   ├── index.py                    # FAISS build/search
│   └── rag.py                      # retrieve → (optional) LLM answer
├── tests/                          # pytest: landmines + data + eval invariants
├── scripts/                        # CLI wrappers (prep, train, eval)
└── requirements.txt
```

Rationale: notebook = the runnable narrative for the grader; `src/` = the same logic as clean, tested, importable modules (this is what "production-grade" honestly means for a take-home — modular, typed, validated, tested code, not distributed infra).

---

## Build categories (sequenced, checkpoint-gated)

Mapped to the assignment's **Suggested Time Allocation**. Each category ends with a gate.

| # | Category | Maps to spec time | Gate |
|---|---|---|---|
| **A** | **Scaffold & env** — repo, `requirements.txt`, config, README skeleton | (setup) | Repo importable, deps pinned |
| **B** | **Data pipeline** — load Itihasa + FLORES + Gita; Unicode NFC; dedup; build directional pairs; dual-script (Devanagari+IAST) aug; train/val/test split | Dataset prep 3–5 h | Clean pair counts printed; splits disjoint; tests green |
| **C** | **Baseline eval** — base e5-small IR metrics (in-domain + OOD, both directions) | (part of eval) | Baseline numbers recorded |
| **D** | **Training** — MNRL fine-tune (run A); optional hard-negatives + Cached-MNRL (run B) | Training 3–6 h | Fine-tuned checkpoint saved; loss curve sane |
| **E** | **Evaluation & analysis** — before/after table, charts, failure taxonomy, IAST robustness | Eval/analysis 2–4 h | Before/after deltas + ≥10 qualitative cases |
| **F** | **RAG demo** — FAISS index + retrieve top-k + optional grounded LLM answer | (product) | End-to-end query→answer works |
| **G** | **Docs** — README, REPORT.md (9 spec sections), model-choice comparison ("why e5, not bge-m3") | Documentation 1–2 h | All 9 report sections present |
| **H** | **Bonus (ranked, if time)** — see below | (stretch) | Each bonus self-contained |
| **R** | **Review & test pass** — static review + pytest + notebook dry-run; fix all issues | (final) | All tests pass, notebook runs top-to-bottom on Colab |

**Checkpoint policy:** sanity-check after **B** (data looks right) and after **E** (results are real) — those are the points where a wrong assumption is expensive.

---

## Iteration protocol ("loop engineering", right-sized)

Depth lives in the **experimental loop**, not process ceremony. Target **5–7 logged iterations** on the model:

```
baseline → train → eval(in-domain + OOD) → FAILURE ANALYSIS → adjust → repeat
```
Adjustment levers, in priority order: (1) batch size ↑ for more in-batch negatives, (2) epochs / LR / warmup, (3) dual-script augmentation ratio, (4) hard-negative mining, (5) e5-base vs e5-small, (6) loss variant (Cached-MNRL). Every iteration is a row in a results table in `REPORT.md` — that table *is* the "debug and iterate" evidence the rubric grades.

> Scoping note: depth here means **depth of justification** — per-decision *why* + a rejected alternative + failure analysis (as applied throughout [`03`](03-technical-design.md)) — rather than added process layers that carry no grading payoff.

---

## Deliverables checklist (1:1 with spec "Submission Requirements")

| Spec requirement | Where it's satisfied |
|---|---|
| 1. GitHub repo or zip | The `docs/sanskrit-assignment/` bundle |
| 2. Colab notebook(s) | `notebooks/sanskrit_retrieval_finetune.ipynb` |
| 3. README with setup | `README.md` |
| 4. Short report (PDF/MD) | `report/REPORT.md` — 9 sections |
| 5. Sample outputs | Qualitative retrievals + RAG answers in notebook & report |
| 6. Trained adapters/checkpoints (optional) | Saved to disk / HF Hub; link in README |
| 7. Evaluation scripts | `src/…/evaluate.py` + `scripts/` |

**Report's 9 required sections** (spec): problem understanding · dataset prep · why this base model · fine-tuning approach · hardware constraints & optimizations · evaluation methodology · failure cases · challenges · what I'd improve with more time. The "why this, not that" comparison lives in *§3 (base model)* + a dedicated comparison table.

---

## Risk register (with fallbacks)

| Risk | Likelihood | Fallback |
|---|---|---|
| FLORES `san_Deva` gated/unavailable | Med | Use IN22 Sanskrit or a held-out Itihasa OOD slice |
| Gita dataset license unclear | Low | Use only public-domain source; drop from redistribution, keep for local demo |
| Colab OOM at large batch | Low | Drop batch, use Cached-MNRL for effective batch; seq-len 128 |
| In-domain metrics look inflated | Med | Lead with the IN22/FLORES OOD number; disclose domain skew |
| Transliteration mismatch tanks IAST queries | Med | Dual-script augmentation (already in plan B); test both scripts |
| Base model already strong → small delta | Low | Expected weak Sanskrit baseline; if not, frame as "strong multilingual prior" finding |
| Time overrun | Med | Cut order: bonus (H) → run B → e5-base. Core A–G always ships |

---

## Bonus stretch (ranked by effort-to-impact)

1. **Base-vs-fine-tuned + e5-small-vs-e5-base** grid — high impact, low effort (reuses eval harness).
2. **Tokenizer/Devanagari fertility experiment** — spec explicitly cares; medium/low effort.
3. **INT8 quantization** of the embedder + recall-retention check — medium; ties to Product Thinking.
4. **Hard-negative mining** run — medium effort, real quality signal.
5. **Thematic RAG eval** (hand-labeled ~20–30 queries) — medium; guards against over-claiming.
6. **Error-visualization** (embedding-space UMAP before/after) — low/medium, strong for the report.

---

## 12-point review of this roadmap

Each point is checked against the plan. Full production design is in [`05`](05-production-system-design.md); here is the **scope decision** for the take-home.

| # | Perspective | Take-home scope decision | Prod design |
|---|---|---|---|
| 1 | **Scaling** | N/A — single-node notebook; index fits RAM (~285MB). Correct for scope. | [05 §1] |
| 2 | **Optimization** | `IndexFlatIP` (exact) for honest recall; seq-len 128 + fp16 + batch tuning. INT8 = bonus. | [05 §2] |
| 3 | **Performance metrics** | Offline Recall@K/MRR/nDCG (in-domain+OOD) + train time + index size. | [05 §3] |
| 4 | **Request handling** | RAG demo validates input, applies prefixes server-side, top-k timeout. | [05 §4] |
| 5 | **Vulnerability** | Input validation on query; **prompt-injection note** for the LLM RAG step; pinned deps; verify HF model provenance. | [05 §5] |
| 6 | **SDLC** | Repo structure, pinned reqs, pytest, **eval-as-a-gate** in review pass R, reproducible seeds. | [05 §6] |
| 7 | **Compliance** | **Strength:** Itihasa public-domain + e5 MIT = clean licensing. Respectful handling of sacred text noted. | [05 §7] |
| 8 | **DR & backup** | **Index is derivable** from source corpus + model → cheap rebuild; checkpoint saved. | [05 §8] |
| 9 | **Support & helpdesk** | N/A for take-home; feedback→hard-negative flywheel designed. | [05 §9] |
| 10 | **System maintenance** | Retraining cadence + re-embedding on model upgrade + drift monitoring (designed). | [05 §10] |
| 11 | **Network & security** | N/A single-node; TLS/authN/VPC/egress-control designed. | [05 §11] |
| 12 | **Resource mgmt / automation** | Scripts + config-driven runs (no hardcoded values); IaC/autoscaling designed. | [05 §12] |
| 13 | **Other** (observability, ingestion-at-scale, transliteration-in-prod, online eval, data flywheel) | Transliteration normalization is *in* the take-home; rest designed. | [05 §13] |

**Verdict:** the roadmap is coherent and correctly scoped — take-home builds the model + eval + demo + tests; the 12 points are satisfied as *design* where building them would be out-of-scope over-engineering. No blocking issues found in review.

---

## Execution plan
Execute **A → G** category by category (pausing at the B and E checkpoints), then the **R** review+test pass (static review + pytest + notebook dry-run) until clean. Training runs on Colab (T4/L4).
