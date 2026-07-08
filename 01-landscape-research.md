# Phase 1 — Comparative Landscape Research

**Assignment:** AI/ML Model Training take-home — Sanskrit + English NLP.
**Constraints (ground truth from spec):** single Google Colab **T4 (16GB) / L4 (24GB)**, **1–2 days**, one track only. Graded on approach, tradeoffs, evaluation rigor, and communication — *not* SOTA metrics.
**Status:** Research only. No track committed, no data downloaded, no training run. Verdicts below feed the Phase 2 decision memo.

> Method note: the spec's suggested base models (Llama-3.2-1B, Gemma-2B, bge-small, mT5, etc.) are treated as *illustrative, 2024-era* pointers, not authoritative. All model/dataset claims below were checked against the current (2026) Hugging Face hub, MTEB/leaderboards, and recent papers. Where a fact could not be verified live, it is flagged.

---

## Option 1 — Fine-tune a small multilingual LLM for Sanskrit↔English instruction following

Tasks in scope: Sa→En / En→Sa translation, verse explanation, QA over Sanskrit text, simple grammar/morphology reasoning, summarization.

### Current best-fit base models (2026)

The decisive factor is **tokenizer fertility on Devanagari** — a poor tokenizer shatters each Sanskrit word into 3–5 subwords, burning context and weakening the learning signal per example.

| Model | HF repo | Params | License | Context | Devanagari tokenizer | Notes |
|---|---|---|---|---|---|---|
| **Sarvam-1** | `sarvamai/sarvam-1` | ~2B | **Non-commercial** (custom) | 8,192 | **Best: fertility ~1.4–2.1 tok/word**, purpose-built 68K Indic vocab | Strongest Indic fit, but a **base model** (no chat template — you build it) and non-commercial |
| **Gemma 3 1B** | `google/gemma-3-1b-it` | 1B | Gemma (commercial OK) | 32K | Moderate; 262K vocab rebalanced for complex scripts | Instruction-tuned OOTB, trivial QLoRA on T4; **1B is English-focused** — multilingual depth thin |
| **Gemma 3 4B** | `google/gemma-3-4b-it` | 4B | Gemma (commercial OK) | 128K | Same 262K vocab, **officially multilingual (140+ langs)** | Best Gemma quality; 4B QLoRA is the **upper edge of a 16GB T4** — prefer L4 |
| **Qwen2.5-1.5B-Instruct** | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | **Apache-2.0** | 128K | Devanagari **not an official target lang** → high fertility | Cleanest license + strong instruction-following, weakest Devanagari coverage |
| **Llama-3.2-1B / 3B-Instruct** | `meta-llama/Llama-3.2-{1B,3B}-Instruct` | 1B / 3B | Llama Community (gated, MAU cap) | 128K | Hindi supported but high fertility on Sanskrit sandhi | The spec default; fine baseline, mediocre Devanagari efficiency |

**Not the fine-tune target, but useful:** `ai4bharat/indictrans2-en-indic-1B` is a translation-only NMT model that **officially supports Sanskrit (`san_Deva`)** — best used as a **data-generation / back-translation tool and a translation-quality reference baseline**, not the instruction model. Airavata (Hindi-only) and Sarvam-M/large (too big for T4/L4) are out.

**T4/L4 reality:** 1B–2B QLoRA is comfortable on a 16GB T4; 3B fits with 4-bit + short sequences; 4B is tight on T4, comfortable on L4.

### Dataset reality check

Central truth: **almost all Sanskrit data is raw text or translation pairs, not instruction-response pairs — you must synthesize the instruction set.**

- **`rahular/itihasa`** *(verified)* — **~93,033 Sa–En verse pairs** (75,162 train / 6,149 val / 11,722 test), Apache-2.0. Rāmāyana + Mahābhārata shlokas + English. **Backbone for translation + verse-explanation.** Domain-narrow (archaic epic register).
- **Sāmayik** *(github.com/ayushbits/saamayik, arXiv:2305.14004)* — ~53K En↔Sa **contemporary prose** pairs. Complements Itihāsa's archaic verse (verify license).
- **`snskrt/*` HF collection** — large raw-text corpora: Vyakaran 616K (grammar), Sanskrit_shlokas 194K, Mahabharat 148K, etc. Mostly **raw text** → seed material for synthetic generation + grammar tasks.
- **AI4Bharat IndicCorp / OPUS Sa–En / Wikipedia sa** — small, noisy, raw text; supplementary only. OPUS Sa–En clean count could not be verified.

**The real work is synthetic instruction construction:** template Itihāsa/Sāmayik into translation instructions (free, thousands of reliable pairs); teacher-LLM-generate explanation/QA/summarization over verses + gloss and spot-check (no ground-truth dataset exists for these); seed grammar/morphology from Vyakaran (hardest to auto-verify — keep small, mark experimental). Realistic target: **2,000–5,000 pairs**, ~60% templated translation, ~40% teacher-generated + reviewed.

### Known hard problems
- **Sandhi/compounding → subword fragmentation** (why Sarvam's low fertility matters; Llama/Qwen worse).
- **Devanagari ↔ IAST/Harvard-Kyoto/SLP1 variation** — normalize to one canonical script (Devanagari) on day 1 or the model learns transliteration noise.
- **Tiny-model hallucination** on low-resource languages — fluent-but-wrong translations/etymologies.
- **Catastrophic forgetting** — largely mitigated by QLoRA (frozen base) + a little English data mixed in.
- **No standard Sanskrit instruction benchmark** — Itihāsa test split is the only ready-made held-out *translation* set; explanation/QA eval must be bespoke.

### Realistic 1–2 day scope
QLoRA fine-tune of a 1–3B model on 2–5K synthetic pairs, full pipeline data→train→eval→write-up. **Data prep 4–6h** (dominant cost — teacher generation + review), **training 1.5–4h** (T4, 1–2B), **eval 2–3h**, **write-up 2h**. **Cut:** custom tokenizer work; rigorous grammar/morphology; full test-set eval (sample 200–500); multi-model bake-offs.

### Evaluation feasibility
**chrF++ (lead) + BLEU** via `sacrebleu` on 200–500 held-out Itihāsa pairs; **LLM-as-judge** (fixed rubric, 30–50 prompts) for explanation/QA/summary; base-vs-QLoRA **before/after** with 5–8 side-by-side examples. Realistic, but absolute scores will be low — frame the win as *relative improvement + methodology*.

### Feasibility verdict: **GREEN** (with a **YELLOW** asterisk on absolute quality)
Clear path, verified data, QLoRA fits T4/L4, credible before/after. Asterisk: output quality will be modest and grammar reasoning shallow — inherent to the domain.

**Open risks**
- **License trap:** best-tokenizer Sarvam-1 is **non-commercial**; Llama is gated + MAU-capped. If commercial use is implied, default to **Gemma-3** or **Qwen2.5 (Apache-2.0)**.
- **Synthetic instruction quality** — teacher-generated "ground truth" can inflate eval; spot-check and disclose.
- **Tokenizer fragmentation** on non-Sarvam models caps quality; unfixable in-budget.
- **Script normalization** must be decided day 1.
- **Sarvam-1 is a base (non-chat) model** — budget ~1h to define the prompt template.
- **Domain narrowness** (Itihāsa = archaic verse) — mix in Sāmayik prose.
- **BLEU misleads** for morphologically rich Sanskrit — lead with chrF + qualitative.

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
| **Bhagavad Gita** (`JDhruv14/…`, `vedicscriptures/bhagavad-gita`) | **701 verses** | Devanagari + IAST + English + Hindi | source PD (verify card) | Clean eval/demo; free IAST↔Devanagari pairs |
| **FLORES-200/+** (`openlanguagedata/flores_plus`) | **`san_Deva`: 997 dev + 1012 devtest** | Professional, sentence-aligned | CC-BY-SA-4.0 | **Gold held-out eval** (out-of-domain, honest before/after) |
| **IN22** (`ai4bharat/IN22-Gen`,`-Conv`) | ~1,024 n-way incl. Sanskrit | Professional | Open | Second eval domain |
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
Cheapest of the three. **Data prep 2–4h**, baseline eval ~30min, **train run A** (MultipleNegativesRankingLoss, e5-small, ~80K pairs) **min–1h/epoch on T4**, optional run B (+2–4h), **FAISS index + RAG demo 1–2h**, **eval + write-up 1–2h**. FAISS-flat is fine at ~90K docs; no GPU index or quantization needed.

### Evaluation feasibility
**The track's differentiator.** Parallel verses give free 1:1 relevance labels → **quantitative, reproducible before/after with zero annotation**. Report **Recall@1/5/10, MRR@10, nDCG@10** via `sentence-transformers`' `InformationRetrievalEvaluator`, base vs fine-tuned, on (a) in-domain Itihāsa held-out and (b) **FLORES/IN22** (out-of-domain, honest number). Weak baseline → dramatic in-domain delta; smaller real gain on FLORES.

### Feasibility verdict: **GREEN**
Cheap training, abundant clean aligned data, XLM-R bases that genuinely cover Sanskrit, and label-free quantitative before/after. Most tractable and most cleanly measurable track on a single T4/L4 in 1–2 days.

**Open risks**
- Verify **Devanagari tokenizer coverage** before gte-multilingual / arctic-`m`; default to XLM-R-lineage (e5, bge-m3).
- **License hygiene:** use MIT/Apache; **avoid jina-v3 (CC-BY-NC)**; confirm E5 LICENSE line.
- **Transliteration:** dual-script augmentation mandatory if IAST queries expected.
- **Domain skew:** archaic/OCR-noisy Itihāsa English may inflate in-domain — lean on FLORES/IN22 as honest held-out.
- **Eval framing:** 1:1 verse retrieval is *translation retrieval* — state clearly; optionally hand-label ~20–30 thematic queries for a RAG-realistic eval.
- **Silent landmines:** E5 prefixes + consistent L2-normalization must be baked into train *and* eval.

---

## Option 3 — Post-OCR correction model for Sanskrit/Indic text

Tasks in scope: seq2seq fine-tuning (noisy OCR Devanagari → corrected Devanagari), synthetic OCR-noise generation, Unicode/script normalization.

### Current best-fit base models (2026)

| Model | HF repo | Params | License | Tokenization | Devanagari fit |
|---|---|---|---|---|---|
| **ByT5-small** | `google/byt5-small` | ~300M | Apache-2.0 | **Byte-level** (tokenizer-free UTF-8) | **Best fit** — handles matras/nukta/ZWJ natively, no OOV, robust to noisy spelling |
| **mT5-small** | `google/mt5-small` | ~300M | Apache-2.0 | SentencePiece subword | Good; ~4× shorter sequences than byte-level; subwords brittle on corrupted conjuncts |
| **mT5-base** | `google/mt5-base` | ~580M | Apache-2.0 | Subword | Comfortable on L4, tight-but-trainable on T4 |
| **IndicBART** | `ai4bharat/IndicBART` | ~244M | MIT | Subword, **Devanagari-native** | Purpose-built for Indic; smallest/fastest; strong Devanagari prior |

**Byte vs subword tradeoff:** Devanagari is an abugida (conjunct ligatures, attaching matras, combining nukta/ZWJ). Subword tokenizers mis-merge these under noise; ByT5 sees a one-glyph OCR error as a **local, learnable byte edit**. The Sanskrit post-OCR benchmark's best system is **ByT5 + SLP1 phonetic encoding** (Maheshwari et al., EMNLP-Findings 2022), and the 2026 Devanagari stress-test also chose ByT5-small. **Cost:** each Devanagari char is 3 UTF-8 bytes → **3–5× longer sequences**, quadratic in attention. Mitigate on T4: cap `max_length` 256–512 bytes (sentence/line granularity), batch 4–8, fp16 + grad-accum. ByT5-*small* trains fine on T4; ByT5-base wants L4.
**Head start:** `chronbmm/sanskrit-byt5-*` hosts ByT5 already adapted to Sanskrit (ByT5-Sanskrit, arXiv:2409.13920) — a legitimate, citable starting point over vanilla `byt5-small`.

### Dataset reality check
**Advantage: you can bootstrap unlimited pairs by corrupting clean text.** Clean Devanagari Sanskrit is abundant.

| Source | Access | Size | Script | License |
|---|---|---|---|---|
| **Sanskrit Wikipedia** (sa.wikipedia) | Full Wikimedia dump | ~12K articles | **Devanagari** | CC BY-SA — cleanest for redistribution |
| **sanskritdocuments.org** | Per-doc download | Thousands | **Devanagari** | Free non-commercial + attribution |
| **GRETIL** | Bulk ZIP | Hundreds of texts | Mostly **romanized** (HK/IAST) | Per-text academic terms |
| **Digital Corpus of Sanskrit** | Downloadable, tagged | ~650K+ tagged sentences | Romanized | Academic; cite |
| **Sanskrit Library** | Web + some download | Large curated | Devanagari + romanized | Academic; not fully open bulk |

**Synthetic OCR-noise pipeline — strong prior art:**
- **RoundTripOCR** (arXiv:2412.15248) — render clean text in 50 Devanagari fonts → Tesseract → pair OCR output with source ("OCR errors as mistranslations"). Released **4.07M Sanskrit pairs**. Gold-standard realistic route (needs Tesseract + font rendering).
- **Hand-built corruption** (confusion tables, matra drop/swap, ligature-break, Unicode confusables, spacing) — faster to code but transfers poorly to real scans; use as *supplement*.

**Real OCR-error paired Sanskrit data (verified public):**
- **Maheshwari et al. 2022** (ACL Findings; arXiv:2211.07980) — **~218K sentences / 1.5M words from 30 books**, real OCR paired with ground truth, on GitHub. **The single most important asset — a genuine real-OCR held-out test set.** Best baseline ByT5+SLP1: **+23 points** over raw OCR.

### Known hard problems
- **Realistic vs random noise (#1 failure mode).** The 2026 stress-test: correctors **do not transfer across OCR engines** (+1.2–1.5 chrF++ on matched engine, ~0 elsewhere), and "synthetic renders badly overstate quality" (EasyOCR chrF++ 93.6 synthetic → 58.3 real). Random char-flips overfit to noise real scanners never produce.
- **Unicode normalization** — NFC/NFD ambiguity, matra ordering, nukta precomposed vs base+U+093C, ZWJ/ZWNJ. Without deterministic NFC + matra-order canonicalization, CER is inflated by invisible codepoint diffs and capacity is wasted. (SLP1/romanization flattens this into unambiguous ASCII.)
- **Conjunct/ligature reconstruction** (क्ष, ज्ञ, त्र split/merged by OCR).
- **Over-correction** — a fluent LM "improves" already-correct archaic/technical text. Must measure **CER on clean inputs** (should stay ≈0).
- **Byte-level sequence blowup** — the binding T4 constraint.

### Realistic 1–2 day scope
- **Data-gen pipeline 0.5–0.75 day — the real work.** Pull clean Devanagari (Wikipedia + sanskritdocuments); build corruption fn (hand table + optional light RoundTripOCR render→Tesseract subset); NFC-normalize both sides.
- **Fine-tune ByT5-small** (sentence-level, `max_length` 256–512, fp16, 1–3 epochs, tens of thousands of pairs) — few hours on T4; optionally IndicBART/mT5-small as subword comparison.
- **CER/WER before-vs-after on held-out synthetic** — headline.
- **Stretch (high value):** evaluate on a slice of **real** Maheshwari-2022 OCR to expose the synthetic→real gap honestly.
- **Cut:** real scanned-PDF→Tesseract→correction over many docs; multi-engine robustness; ByT5-base/large; RoundTripOCR-scale font rendering.

### Evaluation feasibility
Before/after **CER & WER** (`jiwer`/`evaluate`) on held-out synthetic is easy; also report **CER on clean inputs** (over-correction). **Mandatory honesty caveat:** synthetic-noise eval overstates real-world gains (matched-distribution gains largely vanish OOD). The most defensible result is the **contrast between strong synthetic CER improvement and the real-OCR-set CER** — that gap *is* the finding.

### Feasibility verdict: **GREEN**
Synthetic-corruption pipeline + ByT5-small + CER/WER before/after fits T4/L4 in 1–2 days, well de-risked by recent prior art with citable baselines.

**Open risks**
- **Synthetic→real transfer gap** — dominant scientific risk; synthetic gains may not reflect real scans. Mitigate with a render→Tesseract subset + real-data eval slice; frame honestly.
- **Unicode normalization bugs** — inconsistent NFC/matra-order silently inflates CER and poisons training.
- **Byte-level sequence/memory blowup on T4** — keep ByT5-*small*, cap length, fp16 + grad-accum.
- **Over-correction** on archaic/technical vocabulary — measure clean-input CER; consider SLP1 target space.
- **License hygiene** — GRETIL/DCS/Sanskrit Library are academic-terms + largely romanized; Wikipedia (CC BY-SA) + sanskritdocuments are cleanest Devanagari.
- **Time sink is data-gen, not training** — budget the corruption pipeline as the main engineering effort.

---

## Cross-option summary

| Dimension | Option 1 (LLM instruct) | Option 2 (Embeddings) | Option 3 (Post-OCR) |
|---|---|---|---|
| Feasibility verdict | 🟢 (🟡 on quality) | 🟢 | 🟢 |
| Training cost on T4/L4 | Medium (QLoRA 1–3B) | **Low** (small ST model) | Medium (byte-level seq blowup) |
| Clean ready-to-use data | Partial — must synthesize instructions | **Abundant aligned (~93K)** | Clean text abundant; **pairs are synthetic** |
| Quantitative before/after | Yes, but noisy (chrF + LLM-judge) | **Yes, clean & label-free** (Recall@K/MRR/nDCG) | Yes, but **synthetic→real gap** undermines credibility |
| Dominant risk | License traps + synthetic-instruction quality | Transliteration + eval framing | Synthetic noise not transferring to real scans |
| Biggest work chunk | Instruction data synthesis | (evenly small) | OCR-noise data-gen pipeline |

All three are feasible. They differ most on **evaluation cleanliness** and **where the effort concentrates** — the axes the Phase 2 decision memo scores.

---

### Sources
Sarvam-1 · rahular/itihasa · Sāmayik (arXiv:2305.14004) · IndicTrans2 (san_Deva) · Gemma 3 · Qwen2.5-1.5B · Llama-3.2 · snskrt datasets · IndicSuperTokenizer (arXiv:2511.03237) · multilingual-e5 (small/base) · BAAI/bge-m3 · gte-multilingual-base · snowflake-arctic-embed-v2 · FLORES+ (`san_Deva`) · AI4Bharat IN22/BPCC · google/byt5-small · google/mt5-small · ai4bharat/IndicBART · RoundTripOCR (arXiv:2412.15248) · Maheshwari et al. EMNLP-F 2022 (arXiv:2211.07980) · Devanagari OCR stress-test 2026 (arXiv:2606.29213) · ByT5-Sanskrit (arXiv:2409.13920).
