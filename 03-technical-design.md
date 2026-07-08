# Phase 3 — Technical Design (Option 2: Sanskrit↔English Retrieval)

**Track:** Fine-tune a multilingual embedding model for Sanskrit↔English semantic retrieval (contrastive learning + mini-RAG demo + Recall@K/MRR/nDCG).
**Compute:** Colab **T4 (16GB)** primary; **L4 (24GB)** if available; local RTX 3050 for dev/smoke-tests only (VRAM-limited → hurts in-batch negatives — see [`01`](01-landscape-research.md)).
**Grounding:** [`01-landscape-research.md`](01-landscape-research.md) + [`02-decision-memo.md`](02-decision-memo.md) + assignment spec.

Every decision below states *what* and *why*, with at least one **rejected alternative**.

---

## 1. Base model

**Decision: `intfloat/multilingual-e5-small`** (primary), with **`intfloat/multilingual-e5-base`** as the optional quality run.

**Why:**
- **XLM-RoBERTa lineage → genuinely covers Sanskrit.** CC-100 training data includes `sa` (~44M tokens); Devanagari is in the SentencePiece vocab, tokenized as subwords with **no `[UNK]` explosion**. It has *seen* Sanskrit at low resource → **weak baseline + large fine-tuning headroom**, which makes the before/after dramatic (good for the report).
- **Cheap on T4.** 118M params, 384-dim, 512 max-seq → full fine-tune <1h/epoch, big batches fit → more in-batch negatives → better contrastive learning.
- **Clean license (MIT)** → no compliance friction (contrast Option-1's Sarvam non-commercial trap).
- **Asymmetric `query:`/`passage:` prefixes** built for exactly this cross-lingual query→doc task.

**Rejected alternatives:**
| Model | Why rejected |
|---|---|
| `BAAI/bge-m3` (568M) | Borderline on 16GB T4; 8192-context is overkill for short verses; slower iteration. Kept as a *stretch* comparison only. |
| `Alibaba-NLP/gte-multilingual-base` | From-scratch encoder, **Sanskrit not listed**, Devanagari tokenizer coverage unverified → correctness risk not worth taking in a 1–2 day box. |
| `jinaai/jina-embeddings-v3` | **CC-BY-NC** — non-commercial license disqualifies it for a reusable deliverable. |
| IndicSBERT / MuRIL-based | **Sanskrit not among the 17 supported Indian languages**; monolingual Sanskrit MLMs have no English side → can't do cross-lingual retrieval. |

**Tokenizer note (assignment explicitly cares):** I will *measure* e5's Devanagari fertility (tokens/word) and report it, but **not** retrain the tokenizer — for a 118M model on <100K pairs, vocab surgery is high-risk/low-reward in-budget. Rejected: tokenizer extension (documented as out-of-scope with rationale).

---

## 2. Dataset plan

All chosen from the assignment's suggested Option-2 list; see the comparison in [`02-decision-memo`](02-decision-memo.md) and dataset reasoning in [`01`](01-landscape-research.md).

| Dataset | Role | Size | License / access |
|---|---|---|---|
| **`rahular/itihasa`** (= "Sanskrit-English aligned corpora") | **Primary train + in-domain test** | ~75K train / 6K val / 12K test aligned Sa↔En verse pairs | Public domain |
| **Bhagavad Gita** (`JDhruv14/Bhagavad-Gita_Dataset`) | **RAG demo + qualitative eval**; `sanskrit`+`english` cols | 701 verses | Public-domain source; **not gated** |
| **AI4Bharat IN22** (`ai4bharat/IN22-Gen`, cfg `all`, split `gen`) | **Out-of-domain eval** (`sentence_san_Deva`/`sentence_eng_Latn`) | ~1,024 n-way | Open license; **gated → needs HF auth** |
| **FLORES-200 `san_Deva`** (`openlanguagedata/flores_plus`) | OOD eval **fallback** if IN22 auth unavailable | 997 dev + 1012 devtest | CC-BY-SA-4.0; gated |
| ~~Upanishads / OPUS~~ | **Deliberately skipped** | — | sparse/uneven alignment, low ROI in 1–2 days |

**Pair construction:**
- Directional training pairs: `(query: <Sanskrit verse>, passage: <English translation>)` **and** the reverse, so retrieval works both directions.
- **No synthetic positives needed** — alignment is 1:1 and real. This is the track's data advantage.
- **Transliteration augmentation (key):** duplicate a fraction of Sanskrit-side text into **IAST** via `indic-transliteration`, so the model is robust to IAST queries (graders may query in Roman script). Mandatory, not optional.
- **Hard negatives (run B):** mine via base-model top-k (`bonus.mine_triplets`), excluding the gold and near-duplicate verses (range-min skip to avoid false negatives).

**Splits:** train on Itihasa `train`; validate on Itihasa `val`; **in-domain test = Itihasa `test`** (held out from training); **OOD test = IN22** (FLORES fallback; Itihasa-slice proxy if neither is authorized). This two-domain eval is the honesty guard against overfitting to archaic epic English.

**Known gap:** Itihasa English is 19th-c. archaic prose with residual OCR noise → in-domain metrics may be optimistic; the IN22/FLORES OOD number is the trustworthy one. Documented explicitly. Both OOD sources are gated, so an unauthenticated run reports the labelled Itihasa-slice proxy instead.

---

## 3. Training method

**Decision: full fine-tune of e5-small with `MultipleNegativesRankingLoss` (MNRL, in-batch negatives), fp16, largest batch VRAM allows.**

**Why:**
- MNRL is the standard, strongest-for-effort contrastive objective for parallel-pair data; **in-batch negatives** are free and effective.
- **Batch size is the real lever** — more in-batch negatives = better embeddings. On T4 → batch 64–128. (Note: gradient accumulation does *not* increase in-batch negatives → prioritize raw batch; use `CachedMultipleNegativesRankingLoss` in run B to fake a large effective batch.)
- Full FT (not LoRA) because 118M is small and full FT is both feasible and better here; LoRA on an embedding model of this size buys little.

**Compute math (T4, 16GB):** e5-small (~0.24GB fp16 weights) + optimizer + activations at seq-len 128 (verses are short), batch 64 → comfortably fits. ~75K pairs × 1–3 epochs ≈ minutes to ~1h. Leaves ample budget for eval + a second run.

**Rejected alternatives:** TripletLoss with explicit triplets (more data-engineering for no gain over MNRL on this corpus); LoRA/QLoRA (unnecessary at 118M); full-precision fp32 (wastes VRAM → smaller batch → worse negatives).

**Iteration protocol (the "loop"):** baseline eval → train run A → eval → **failure analysis** → adjust (batch size, epochs, LR, add dual-script aug, add hard negatives) → re-run. Target **5–7 experimental iterations** logged in a results table — this depth *is* the deliverable, per the rubric ("ability to debug and iterate").

---

## 4. Evaluation methodology

**Primary (quantitative, label-free):** `sentence-transformers` `InformationRetrievalEvaluator` → **Recall@1/5/10, MRR@10, nDCG@10**, computed on:
- **In-domain:** Itihasa test (corpus = English translations; queries = Sanskrit verses; gold = aligned translation).
- **OOD:** AI4Bharat IN22 (FLORES `san_Deva` fallback) — the honest generalization number.
- **Both directions:** Sa→En and En→Sa.

**Before/after:** identical eval on **base e5-small vs fine-tuned** — the headline result. Weak Sanskrit baseline → large, clean delta.

**Qualitative / failure analysis:** top-k retrievals for ~10 hand-picked queries (including thematic ones like *"what does this say about karma?"*), IAST-vs-Devanagari query robustness, and a short taxonomy of failure modes (near-duplicate confusion, transliteration miss, domain shift).

**RAG demo:** FAISS-flat index over the English corpus → query → top-k → (optional) LLM answer grounded in retrieved passages. Demonstrates the end-to-end retrieval product.

**Honesty framing:** 1:1 verse retrieval is *translation retrieval* — easier than open thematic QA. I'll state this and add a small hand-labeled thematic query set to avoid over-claiming.

---

## 5. Silent-failure landmines (baked into code + tests)
- **E5 prefixes** (`query:` / `passage:`) must be applied in **both** train and eval — forgetting them silently tanks results. → unit test.
- **L2 normalization** consistent between index build and query. → unit test.
- **Script normalization** (Unicode NFC + canonical Devanagari) applied before both training and indexing. → unit test.

---

**Next:** Phase 4 roadmap sequences this into build categories; [`05-production-system-design.md`](05-production-system-design.md) covers the 12-point production/deployment design.
