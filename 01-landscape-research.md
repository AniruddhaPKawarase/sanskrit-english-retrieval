# Phase 1 — Landscape Research (chosen track: Option 2, embedding retrieval)

**Assignment:** AI/ML Model Training take-home — Sanskrit + English NLP.
**Constraints (ground truth from spec):** single Google Colab **T4 (16GB) / L4 (24GB)**, **1–2 days**, one track only. Graded on approach, tradeoffs, evaluation rigor, and communication — *not* SOTA metrics.
**Status:** Research only. No data downloaded, no training run — verdict feeds the [decision memo](02-decision-memo.md).

> This file covers the **chosen track only** (Option 2). The other two tracks the assignment
> offered — Option 1 (instruction-tuning an LLM) and Option 3 (post-OCR correction) — were
> researched and scored too; that material lives in [`alternatives-considered.md`](alternatives-considered.md).

> Method note: the spec's suggested base models (bge-small, mE5, mT5, etc.) are treated as *illustrative, 2024-era* pointers, not authoritative. All model/dataset claims below were checked against the current (2026) Hugging Face hub, MTEB/leaderboards, and recent papers. Where a fact could not be verified live, it is flagged.

---

## Option 2 — Fine-tune a multilingual embedding model for Sanskrit↔English retrieval

Tasks in scope: contrastive/triplet fine-tuning, mini RAG/retrieval demo, Recall@K / MRR / nDCG evaluation.

### Current best-fit base models (2026)
Key architectural fact: the strongest small multilingual embedders are **XLM-RoBERTa-based**, whose SentencePiece vocab (250K) and CC-100 training data **include Sanskrit** (CC-100 `sa` ≈ 44M tokens). So the XLM-R family tokenizes Devanagari as subwords with **no `[UNK]` explosion** and has genuinely *seen* Sanskrit at low resource — real coverage + weak baseline = **large fine-tuning headroom**.

| Model (HF repo) | Params | Dim | Max seq | License | Sanskrit/Devanagari reality | T4/L4 |
|---|---|---|---|---|---|---|
| **`intfloat/multilingual-e5-small`** | ~118M | 384 | 512 | MIT* | MiniLM distilled from XLM-R (100 langs); Devanagari in-vocab; needs `query:`/`passage:` prefixes | **Easiest — top pick.** <1h/epoch on T4 |
| **`intfloat/multilingual-e5-base`** | ~278M | 768 | 512 | MIT* | `xlm-roberta-base`, full Sanskrit coverage; prefixes required | **Yes** (~1–3h). Best quality/effort |
| **`BAAI/bge-m3`** | ~568M | 1024 | 8192 | MIT | `xlm-roberta-large`, 100+ langs, dense+sparse+ColBERT | **Borderline on T4**, fine on L4; long-seq overkill for verses |
| **`Alibaba-NLP/gte-multilingual-base`** | ~305M | 768 | 8192 | Apache-2.0 | **Sanskrit not listed**, from-scratch encoder — Devanagari coverage **unverified** | Verify tokenization first |
| **`Snowflake/snowflake-arctic-embed-l-v2.0`** | ~568M | 1024 (Matryoshka→256) | 8192 | Apache-2.0 | `-l` on XLM-R-large (good); `-m` on gte (same caveat) | `-m` on T4, `-l` needs L4/LoRA |
| `jinaai/jina-embeddings-v3` | ~572M | ≤1024 | 8192 | **CC-BY-NC** | Sanskrit not in either lang tier | **Avoid** (non-commercial) |

*E5's MIT license is widely cited; confirm the LICENSE line on the model card before release.
**Indic-specific models are mostly a trap:** IndicSBERT/l3cube are **MuRIL-based (Sanskrit not among the 17 langs)**; `surajp/albert-base-sanskrit` is a monolingual MLM with no English side. Use general XLM-R multilingual embedders. **MTEB context:** the multilingual board is topped by 8B models (irrelevant here); no MTEB task evaluates Sanskrit, so rank is a general-multilingual proxy only.

**Recommendation:** fine-tune **`multilingual-e5-small`** (fast iteration + before/after headroom); add **`-base`** as the quality run if time allows.

### Dataset reality check — aligned Sanskrit↔English
**There is more than enough clean aligned data; synthetic positive pairs are NOT needed.**

| Source | Real size | Alignment | License | Use |
|---|---|---|---|---|
| **`rahular/itihasa`** | **~93,000** shloka↔English pairs | Verse-level, alignment-checked | Public domain | **Primary training corpus** |
| **Bhagavad Gita** (`JDhruv14/Bhagavad-Gita_Dataset`) | **701 verses** | Devanagari + IAST + English + Hindi | source PD (verify card) | Clean eval/demo; free IAST↔Devanagari pairs |
| **FLORES-200/+** (`openlanguagedata/flores_plus`) | **`san_Deva`: 997 dev + 1012 devtest** | Professional, sentence-aligned | CC-BY-SA-4.0 | **Gold held-out eval** (out-of-domain, honest before/after) |
| **IN22** (`ai4bharat/IN22-Gen`,`-Conv`) | ~1,024 n-way incl. Sanskrit | Professional | Open (gated) | Out-of-domain eval domain |
| **BPCC** (`ai4bharat/BPCC`) | Sanskrit subset count **unverified** | Mined + human seed | CC0 / CC-BY | Bonus training data if slice non-trivial |
| Samanantar / OPUS / Samasamayik | excludes Sanskrit / sparse / wrong pair | — | — | Skip |

**Usable aligned pairs: ~90–95K** (Itihāsa dominant) + ~700 Gita + ~2K clean held-out. Ample for contrastive training with **in-batch negatives**; hard negatives are **mined** (adjacent shlokas, BM25/base-model near-misses), not synthesized. Only worthwhile synthetic step is **transliteration augmentation** (Devanagari↔IAST).

### Known hard problems
- **Cross-lingual direction** (Sanskrit query → English doc) — train directional pairs; E5's asymmetric `query:`/`passage:` prefixes fit this. **Forgetting the prefixes silently tanks results (#1 gotcha).**
- **Transliteration mismatch** — Devanagari vs IAST route to different subword regions; **dual-script augmentation is mandatory** if graders may query in IAST.
- **Hard-negative mining** on 93K — in-batch negatives suffice for a strong first run; mine BM25/top-k for a second, avoiding false negatives from near-duplicate verses.
- **Chunking** — shlokas are already short → they *are* the chunks; 512 tokens is plenty.
- **Normalization** — L2-normalize + cosine, consistent between index build and query.
- **Few labels** — verse alignment is 1:1 → every query has exactly one gold passage → Recall@K/MRR/nDCG computable **with zero manual labeling** (but this measures *translation retrieval*, easier than open QA — note the framing).

### Realistic 1–2 day scope
Cheapest of the three tracks. **Data prep 2–4h**, baseline eval ~30min, **train run A** (MultipleNegativesRankingLoss, e5-small, ~80K pairs) **min–1h/epoch on T4**, optional run B (+2–4h), **FAISS index + RAG demo 1–2h**, **eval + write-up 1–2h**. FAISS-flat is fine at ~90K docs; no GPU index or quantization needed.

### Evaluation feasibility
**The track's differentiator.** Parallel verses give free 1:1 relevance labels → **quantitative, reproducible before/after with zero annotation**. Report **Recall@1/5/10, MRR@10, nDCG@10** via `sentence-transformers`' `InformationRetrievalEvaluator`, base vs fine-tuned, on (a) in-domain Itihāsa held-out and (b) an out-of-domain set (Gita / FLORES / IN22). Weak baseline → dramatic in-domain delta; smaller real gain out-of-domain.

### Feasibility verdict: **GREEN**
Cheap training, abundant clean aligned data, XLM-R bases that genuinely cover Sanskrit, and label-free quantitative before/after. Most tractable and most cleanly measurable track on a single T4/L4 in 1–2 days.

**Open risks**
- Verify **Devanagari tokenizer coverage** before gte-multilingual / arctic-`m`; default to XLM-R-lineage (e5, bge-m3).
- **License hygiene:** use MIT/Apache; **avoid jina-v3 (CC-BY-NC)**; confirm E5 LICENSE line.
- **Transliteration:** dual-script augmentation mandatory if IAST queries expected.
- **Domain skew:** archaic Itihāsa English may inflate in-domain — lean on the out-of-domain set as the honest number.
- **Eval framing:** 1:1 verse retrieval is *translation retrieval* — state clearly; optionally hand-label ~20–30 thematic queries for a RAG-realistic eval.
- **Silent landmines:** E5 prefixes + consistent L2-normalization must be baked into train *and* eval.

---

### Sources
multilingual-e5 (small/base) · BAAI/bge-m3 · gte-multilingual-base · snowflake-arctic-embed-v2 · jina-embeddings-v3 · rahular/itihasa · Bhagavad Gita (JDhruv14) · FLORES+ (`san_Deva`) · AI4Bharat IN22/BPCC · IndicSuperTokenizer (arXiv:2511.03237). Options 1 & 3 research + sources: [`alternatives-considered.md`](alternatives-considered.md).
