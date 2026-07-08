# Alternatives Considered — Options 1 & 3 (not chosen)

The assignment offered three tracks. This project implements **Option 2** (multilingual embedding
fine-tuning for Sanskrit↔English retrieval); its research is in [`01-landscape-research.md`](01-landscape-research.md)
and the decision in [`02-decision-memo.md`](02-decision-memo.md). This file preserves the landscape
research and scoring for the two tracks that were evaluated but **not** chosen, so the decision is
auditable.

**Verdict (full matrix below):** Option 2 = **48/50**, Option 1 = **38**, Option 3 = **36**.

---

## Option 1 — Fine-tune a small multilingual LLM for Sanskrit↔English instruction following

Tasks in scope: Sa→En / En→Sa translation, verse explanation, QA over Sanskrit text, simple grammar/morphology reasoning, summarization.

### Current best-fit base models (2026)

The decisive factor is **tokenizer fertility on Devanagari** — a poor tokenizer shatters each Sanskrit word into 3–5 subwords, burning context and weakening the learning signal per example.

| Model | HF repo | Params | License | Context | Devanagari tokenizer | Notes |
|---|---|---|---|---|---|---|
| **Sarvam-1** | `sarvamai/sarvam-1` | ~2B | **Non-commercial** (custom) | 8,192 | **Best: fertility ~1.4–2.1 tok/word**, purpose-built 68K Indic vocab | Strongest Indic fit, but a **base model** (no chat template) and non-commercial |
| **Gemma 3 1B** | `google/gemma-3-1b-it` | 1B | Gemma (commercial OK) | 32K | Moderate; 262K vocab rebalanced for complex scripts | Instruction-tuned OOTB, trivial QLoRA on T4; **1B is English-focused** — multilingual depth thin |
| **Gemma 3 4B** | `google/gemma-3-4b-it` | 4B | Gemma (commercial OK) | 128K | Same 262K vocab, **officially multilingual (140+ langs)** | Best Gemma quality; 4B QLoRA is the **upper edge of a 16GB T4** — prefer L4 |
| **Qwen2.5-1.5B-Instruct** | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | **Apache-2.0** | 128K | Devanagari **not an official target lang** → high fertility | Cleanest license + strong instruction-following, weakest Devanagari coverage |
| **Llama-3.2-1B / 3B-Instruct** | `meta-llama/Llama-3.2-{1B,3B}-Instruct` | 1B / 3B | Llama Community (gated, MAU cap) | 128K | Hindi supported but high fertility on Sanskrit sandhi | The spec default; fine baseline, mediocre Devanagari efficiency |

**Not the fine-tune target, but useful:** `ai4bharat/indictrans2-en-indic-1B` is a translation-only NMT model that **officially supports Sanskrit (`san_Deva`)** — best used as a **data-generation / back-translation tool and a translation-quality reference baseline**, not the instruction model. Airavata (Hindi-only) and Sarvam-M/large (too big for T4/L4) are out.

**T4/L4 reality:** 1B–2B QLoRA is comfortable on a 16GB T4; 3B fits with 4-bit + short sequences; 4B is tight on T4, comfortable on L4.

### Dataset reality check
Central truth: **almost all Sanskrit data is raw text or translation pairs, not instruction-response pairs — the instruction set must be synthesized.**

- **`rahular/itihasa`** — ~93,033 Sa–En verse pairs (75,162 train / 6,149 val / 11,722 test), Apache-2.0.
- **Sāmayik** (github.com/ayushbits/saamayik, arXiv:2305.14004) — ~53K En↔Sa contemporary prose pairs.
- **`snskrt/*` HF collection** — large raw-text corpora (Vyakaran 616K grammar, Sanskrit_shlokas 194K, etc.); mostly raw text → seed for synthetic generation.
- **AI4Bharat IndicCorp / OPUS Sa–En / Wikipedia sa** — small, noisy, raw text; supplementary only.

The real work is **synthetic instruction construction**: template translation instructions; teacher-LLM-generate explanation/QA/summarization (no ground-truth dataset exists); seed grammar from Vyakaran. Realistic target ~2,000–5,000 pairs.

### Known hard problems
- Sandhi/compounding → subword fragmentation. Devanagari ↔ IAST/HK/SLP1 variation → normalize early. Tiny-model hallucination on low-resource text. Catastrophic forgetting (mitigated by QLoRA). No standard Sanskrit instruction benchmark.

### Realistic 1–2 day scope
QLoRA fine-tune of a 1–3B model on 2–5K synthetic pairs. Data prep 4–6h (dominant — teacher generation + review), training 1.5–4h, eval 2–3h, write-up 2h.

### Evaluation feasibility
chrF++ (lead) + BLEU on 200–500 held-out pairs; LLM-as-judge for explanation/QA; base-vs-QLoRA before/after. Absolute scores will be low — frame as relative improvement + methodology.

### Feasibility verdict: **GREEN** (with a **YELLOW** asterisk on absolute quality)
Clear path, verified data, QLoRA fits T4/L4. Asterisk: output quality modest, grammar reasoning shallow — inherent to the domain.

**Open risks:** license traps (Sarvam non-commercial; Llama gated); synthetic-instruction "ground truth" can inflate eval; tokenizer fragmentation caps quality; script normalization must be decided day 1; BLEU misleads for Sanskrit.

### Why it lost (runner-up, score 38)
Richest **model-understanding narrative** (tokenizer fertility, QLoRA, catastrophic forgetting, decoding) and the most impressive-sounding deliverable. It lost on: (1) **noisy evaluation** — no standard Sanskrit instruction benchmark, low absolute scores, teacher-generated "ground truth" is unverifiable; (2) **license traps** — best-tokenizer Sarvam-1 is non-commercial, Llama gated; (3) **slowest to a credible before/after** — riskiest fit for a 1–2 day box. Pick it only to *showcase fine-tuning depth* while accepting a fuzzier eval.

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

**Byte vs subword tradeoff:** Devanagari is an abugida (conjunct ligatures, attaching matras, combining nukta/ZWJ). Subword tokenizers mis-merge these under noise; ByT5 sees a one-glyph OCR error as a local, learnable byte edit. The Sanskrit post-OCR benchmark's best system is ByT5 + SLP1 phonetic encoding (Maheshwari et al., EMNLP-Findings 2022). Cost: each Devanagari char is 3 UTF-8 bytes → 3–5× longer sequences. Head start: `chronbmm/sanskrit-byt5-*` (ByT5-Sanskrit, arXiv:2409.13920).

### Dataset reality check
Advantage: unlimited pairs can be bootstrapped by corrupting clean text. Clean Devanagari Sanskrit is abundant (Sanskrit Wikipedia CC-BY-SA; sanskritdocuments.org; GRETIL; Digital Corpus of Sanskrit; Sanskrit Library).

- **RoundTripOCR** (arXiv:2412.15248) — render clean text in 50 fonts → Tesseract → pair with source; released 4.07M Sanskrit pairs. Gold-standard realistic route.
- **Hand-built corruption** (confusion tables, matra drop/swap, ligature-break, confusables) — faster, transfers poorly; supplement only.
- **Maheshwari et al. 2022** (arXiv:2211.07980) — ~218K sentences / 1.5M words, real OCR paired with ground truth — a genuine real-OCR test set.

### Known hard problems
- **Realistic vs random noise (#1 failure mode):** correctors do not transfer across OCR engines; synthetic renders overstate quality. Unicode normalization (NFC/matra-order/nukta/ZWJ). Conjunct/ligature reconstruction. Over-correction of valid text. Byte-level sequence blowup (binding T4 constraint).

### Realistic 1–2 day scope
Data-gen pipeline 0.5–0.75 day (the real work), fine-tune ByT5-small (CER/WER before-vs-after on held-out synthetic), stretch: evaluate on a real-OCR slice to expose the synthetic→real gap. Cut: full scanned-PDF pipelines, multi-engine robustness.

### Feasibility verdict: **GREEN**
Synthetic-corruption pipeline + ByT5-small + CER/WER before/after fits T4/L4, well de-risked by prior art.

**Open risks:** synthetic→real transfer gap (dominant scientific risk); Unicode normalization bugs; byte-level memory blowup; over-correction; the time sink is data-gen not training.

### Why it lost (score 36)
Strongest pure-engineering story (byte-level modeling, Unicode handling, a synthetic-noise pipeline) and well de-risked. It lost because its **headline metric is scientifically fragile**: the 2026 stress-test shows synthetic-noise gains largely vanish on real scans, so the most honest result is "here's the gap" rather than "here's the win" — a harder story to sell in a time-boxed take-home, and the data-gen pipeline front-loads the effort.

---

## Full three-way decision matrix

Scored 1–5 (5 = best) on the assignment's 8 Common Evaluation Criteria plus 2 added axes.

| Axis | Opt 1 — LLM instruct | Opt 2 — Embeddings | Opt 3 — Post-OCR |
|---|---|---|---|
| Learning Ability | 4 | 4 | 4 |
| Practical ML Engineering | 3 | **5** | 3 |
| Problem Solving | 4 | 4 | 4 |
| Dataset Thinking | 4 | **5** | 4 |
| Model Understanding | **5** | 4 | 4 |
| Evaluation Quality | 3 | **5** | 3 |
| Communication | 4 | **5** | 4 |
| Product Thinking | 4 | **5** | 4 |
| Time-to-credible-result (added) | 3 | **5** | 3 |
| Narrative fit (added) | 4 | **5** | 3 |
| **Total (/50)** | **38** | **48** | **36** |

| Dimension | Option 1 (LLM instruct) | Option 2 (Embeddings) | Option 3 (Post-OCR) |
|---|---|---|---|
| Feasibility | GREEN (yellow on quality) | GREEN | GREEN |
| Training cost on T4/L4 | Medium (QLoRA 1–3B) | **Low** (small ST model) | Medium (byte-level seq blowup) |
| Clean ready-to-use data | Partial — must synthesize instructions | **Abundant aligned (~93K)** | Clean text abundant; **pairs synthetic** |
| Quantitative before/after | Yes, but noisy (chrF + LLM-judge) | **Yes, clean & label-free** (Recall@K/MRR/nDCG) | Yes, but **synthetic→real gap** |
| Dominant risk | License traps + synthetic-instruction quality | Transliteration + eval framing | Synthetic noise not transferring to real scans |

**Conclusion:** all three are feasible; they differ most on **evaluation cleanliness** and **where the effort concentrates**. Option 2 wins on the axes the rubric weights most for an open-ended take-home (Evaluation Quality, Communication, Product Thinking) and is the safest fit for a single T4 in 1–2 days.
