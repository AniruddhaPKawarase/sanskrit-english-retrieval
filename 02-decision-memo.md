# Phase 2 — Decision Memo (track recommendation)

**Selected track:** Option 2 (rationale below).
**Inputs:** [`01-landscape-research.md`](01-landscape-research.md) + assignment spec (8-axis rubric, T4/L4, 1–2 days).

## Scoring method
Each track scored **1–5** (5 = best) on the assignment's own **Common Evaluation Criteria** (8 axes) plus **2 added axes** (Time-to-credible-result, Narrative fit). The rubric grades *how you work and reason*, not final metrics — so axes that reward clean evaluation and clear communication are weighted heavily in the reading, though scores are shown unweighted for transparency.

## Decision matrix

| Axis (what it rewards) | Opt 1 — LLM instruct | Opt 2 — Embeddings | Opt 3 — Post-OCR |
|---|---|---|---|
| **Learning Ability** — picking up unfamiliar tooling | **4** — TRL/PEFT/QLoRA + synthetic-instruction generation is a broad, current stack | **4** — SentenceTransformers + contrastive loss + FAISS; clean, well-documented | **4** — seq2seq + byte-level + an OCR-noise pipeline; the pipeline is the novel learning |
| **Practical ML Engineering** — efficient compute use | **3** — QLoRA 1–3B fits but is the heaviest train; tokenizer fragmentation wastes context | **5** — cheapest by far; small model, <1h/epoch, FAISS-flat, no quantization needed | **3** — byte-level 3–5× sequence blowup is the binding T4 constraint |
| **Problem Solving** — debug & iterate | **4** — many moving parts to debug (data, template, decoding) | **4** — fast iteration loop means more debug cycles in budget | **4** — Unicode/normalization + noise realism are genuinely hard debugging |
| **Dataset Thinking** — quality of data prep | **4** — strong story (synthesize instructions from raw text) but "ground truth" is teacher-generated & unverified | **5** — ~93K clean *aligned* pairs, real held-out (FLORES), mined hard negatives — textbook dataset thinking | **4** — bootstrapping via corruption is clever, but synthetic noise ≠ real distribution |
| **Model Understanding** — architecture/training choice | **5** — richest story: tokenizer fertility, LoRA rank, forgetting, decoding | **4** — bi-encoder, contrastive objectives, negative sampling, normalization | **4** — byte vs subword tradeoff for an abugida is a sharp, defensible argument |
| **Evaluation Quality** — meaningful metrics & analysis | **3** — chrF + LLM-judge works but noisy; low absolute scores; no standard benchmark | **5** — label-free, reproducible Recall@K/MRR/nDCG, clean base-vs-tuned before/after on 2 domains | **3** — CER/WER easy, but synthetic→real gap forces a heavy honesty caveat that dilutes the headline |
| **Communication** — clarity of tradeoffs | **4** — lots to explain, risk of sprawl | **5** — cleanest narrative: query→retrieve→metric, easy charts | **4** — strong if the synthetic-vs-real gap is framed as *the finding* |
| **Product Thinking** — deployment awareness | **4** — instruction models are the obvious product, but a 1–3B Sanskrit model is weak in prod | **5** — retrieval/RAG is a real, shippable production pattern; directly deployable | **4** — post-OCR is a real niche product (digitization pipelines) but narrower market |
| **Time-to-credible-result** *(added)* | **3** — data synthesis + train + fuzzy eval is the slowest to a defensible before/after | **5** — quantitative before/after achievable in well under a day | **3** — data-gen pipeline is front-loaded; result late in the budget |
| **Narrative fit** *(added)* — production-relevant skills for interview | **4** — fine-tuning fluency is valuable | **5** — retrieval + evaluation rigor + RAG thinking are the most transferable, interview-ready skills | **3** — impressive but niche; less transferable narrative |
| **Total (unweighted /50)** | **38** | **48** | **36** |

## Recommendation: **Option 2 — Multilingual embedding fine-tuning for Sanskrit↔English retrieval**

**Why it wins.** It is the only track that gives a **clean, quantitative, label-free before/after** — parallel verses supply 1:1 relevance labels for free, so Recall@K / MRR / nDCG on base vs fine-tuned needs zero manual annotation. That maps directly onto the two axes the rubric weights most heavily for this kind of open-ended take-home (**Evaluation Quality** and **Communication**), where the other two tracks are structurally weaker: Option 1's eval is noisy and its absolute quality low; Option 3's headline metric is undermined by the documented synthetic→real transfer gap. Option 2 is also the **cheapest to train** (small ST model, <1h/epoch on a T4, FAISS-flat — no quantization), which buys more iteration cycles and de-risks finishing inside 1–2 days. Data is a solved problem, not a gamble: **~93K clean aligned Itihāsa pairs** plus **FLORES `san_Deva`** as an honest out-of-domain held-out. And it lands the most transferable, production-relevant story (retrieval + RAG + evaluation rigor).

The research supports this on merits, not on a predetermined answer — and it *also* aligns with my existing production RAG background, which makes the interview conversation stronger. That's a tiebreaker, not the reason.

**Runner-up: Option 1 (LLM instruction tuning) — score 38.** It has the **richest model-understanding narrative** (tokenizer fertility, QLoRA, catastrophic forgetting, decoding) and the most "impressive-sounding" deliverable. It lost on three things: (1) **evaluation is noisy** — no standard Sanskrit instruction benchmark, low absolute scores, and explanation/QA "ground truth" is teacher-generated and unverifiable; (2) **license traps** — the best-tokenizer model (Sarvam-1) is non-commercial and Llama is gated; (3) **slowest to a credible before/after**, the riskiest fit for a 1–2 day box. Choose it only if the goal is to *showcase fine-tuning depth* and you accept a fuzzier eval.

**Option 3 (post-OCR) — score 36.** Strongest pure-engineering story (byte-level modeling, Unicode handling, a synthetic-noise pipeline) and well de-risked by recent prior art. It lost because its **headline metric is scientifically fragile**: the 2026 stress-test shows synthetic-noise gains largely vanish on real scans, so the most honest result is "here's the gap" rather than "here's the win" — a harder story to sell in a time-boxed take-home, and the data-gen pipeline front-loads the effort.

## What the chosen track unlocks (Phase 3)
The Phase 3 technical design (`03-technical-design.md`) covers: final base model + one rejected alternative, final dataset/split plan, tokenizer/script investigation, training-method + compute math, and evaluation methodology.

## Assumptions worth stating
1. **Track: Option 2** (embedding fine-tuning), on the scoring and rationale above.
2. **Licensing:** if a commercial-use requirement applies, it hardens the Option-1 license warning and favours Apache/MIT — Option 2's `multilingual-e5-small` (MIT) is unaffected, so the choice is robust to that constraint.
