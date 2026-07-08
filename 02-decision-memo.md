# Phase 2 — Decision Memo (chosen track: Option 2)

**Selected track:** Option 2 — multilingual embedding fine-tuning for Sanskrit↔English retrieval.
**Inputs:** [`01-landscape-research.md`](01-landscape-research.md) + assignment spec (8-axis rubric, T4/L4, 1–2 days).

## Scoring method
Each track was scored **1–5** (5 = best) on the assignment's **Common Evaluation Criteria** (8 axes) plus **2 added axes** (Time-to-credible-result, Narrative fit). The rubric grades *how you work and reason*, not final metrics — so axes that reward clean evaluation and clear communication carry weight in the reading, though scores are shown unweighted. **Totals: Option 2 = 48/50, Option 1 = 38, Option 3 = 36.** The full three-way matrix and the analysis of the two rejected tracks are in [`alternatives-considered.md`](alternatives-considered.md).

## Option 2 scorecard (the chosen track)

| Axis (what it rewards) | Score | Justification |
|---|---|---|
| **Learning Ability** — picking up unfamiliar tooling | 4 | SentenceTransformers + contrastive loss + FAISS; clean, well-documented stack |
| **Practical ML Engineering** — efficient compute | **5** | Cheapest by far — small model, <1h/epoch, FAISS-flat, no quantization needed on a T4 |
| **Problem Solving** — debug & iterate | 4 | Fast iteration loop means more debug cycles inside the budget |
| **Dataset Thinking** — quality of data prep | **5** | ~93K clean *aligned* pairs, real held-out, transliteration augmentation — textbook dataset thinking |
| **Model Understanding** — architecture/training | 4 | Bi-encoder, contrastive objectives, negative sampling, L2-normalization |
| **Evaluation Quality** — meaningful metrics | **5** | Label-free, reproducible Recall@K/MRR/nDCG; clean base-vs-tuned before/after on two domains |
| **Communication** — clarity of tradeoffs | **5** | Cleanest narrative: query→retrieve→metric, easy charts |
| **Product Thinking** — deployment awareness | **5** | Retrieval/RAG is a real, shippable production pattern; directly deployable |
| **Time-to-credible-result** *(added)* | **5** | Quantitative before/after achievable in well under a day |
| **Narrative fit** *(added)* | **5** | Retrieval + evaluation rigor + RAG thinking are the most transferable, interview-ready skills |
| **Total** | **48 / 50** | clear winner |

## Why Option 2 wins
It is the only track that gives a **clean, quantitative, label-free before/after** — parallel verses supply 1:1 relevance labels for free, so Recall@K / MRR / nDCG on base vs fine-tuned needs zero manual annotation. That maps directly onto the two axes the rubric weights most heavily for an open-ended take-home (**Evaluation Quality** and **Communication**), where the other two tracks are structurally weaker. Option 2 is also the **cheapest to train** (small model, <1h/epoch on a T4, FAISS-flat), which buys more iteration cycles and de-risks finishing inside 1–2 days. Data is a solved problem, not a gamble: **~93K clean aligned Itihāsa pairs** plus a held-out out-of-domain set for an honest number. And it lands the most transferable, production-relevant story (retrieval + RAG + evaluation rigor).

The research supports this on merits, not on a predetermined answer — and it *also* aligns with an existing production RAG background, which strengthens the interview conversation. That is a tiebreaker, not the reason. (Runner-up Option 1 = 38, Option 3 = 36; full reasoning in [`alternatives-considered.md`](alternatives-considered.md).)

## What the chosen track unlocks (Phase 3)
The Phase 3 technical design ([`03-technical-design.md`](03-technical-design.md)) covers: final base model + one rejected alternative, final dataset/split plan, tokenizer/script investigation, training-method + compute math, and evaluation methodology.

## Assumptions worth stating
1. **Track: Option 2** (embedding fine-tuning), on the scoring and rationale above.
2. **Licensing:** if a commercial-use requirement applies, it favours Apache/MIT — Option 2's `multilingual-e5-small` (MIT) is unaffected, so the choice is robust to that constraint.
