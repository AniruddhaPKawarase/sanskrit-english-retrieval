# Sanskrit ↔ English Semantic Retrieval — Fine-Tuning `multilingual-e5`

Fine-tune a small multilingual embedding model for **Sanskrit↔English cross-lingual semantic search**,
with contrastive training, a **FAISS mini-RAG demo**, and a label-free **Recall@K / MRR / nDCG**
before/after evaluation. Built for a single Colab **T4**. (AI/ML take-home — Option 2.)

Full reasoning lives in [`report/REPORT.md`](report/REPORT.md) (also `report/REPORT.pdf`) and the
design docs [`01`](01-landscape-research.md)–[`05`](05-production-system-design.md).

---

## ⚡ Quickstart — run the mini-RAG (clone & run)

```bash
git clone https://github.com/AniruddhaPKawarase/sanskrit-english-retrieval.git
cd sanskrit-english-retrieval
pip install -r requirements.txt          # CPU is fine for the demo (no GPU needed)
python mini_rag.py "what does the Gita say about karma?"
```

First run downloads the embedding model (`multilingual-e5-small`, ~0.5 GB) and the Bhagavad Gita
corpus (701 verses, ~0.7 MB) from Hugging Face, builds a FAISS index, and prints the top matches:

```
[model] intfloat/multilingual-e5-small (BASE — run the notebook for the fine-tuned checkpoint)
[corpus] Bhagavad Gita — downloading + indexing 701 verses ...

Top 5 passages for: 'what does the Gita say about karma?'
------------------------------------------------------------
[1] score=0.33   Arjuna said: Krishna, what is that Brahma, what is Adhyatma, and what is Karma? ...
[2] score=0.31   Arjuna said: O Keshava, what is the description of a man of steady wisdom ...
...
```

More examples:
```bash
python mini_rag.py --k 3 "the duty of a warrior in battle"
python mini_rag.py "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः"      # Sanskrit (Devanagari) query
python mini_rag.py --reverse "detachment from the fruits of action"  # vice-versa: English query -> Sanskrit verses
python mini_rag.py --model artifacts/e5-small-sa-en-mnrl "karma"      # use the fine-tuned checkpoint
```
Both directions work because the model is trained bidirectionally (Sa→En *and* En→Sa pairs):
default retrieves English verses; `--reverse` indexes the Sanskrit verses and retrieves those.

- **Out of the box** the demo downloads the **fine-tuned checkpoint from the Hugging Face Hub**
  (`AniruddhaAI/sanskrit-e5-small-retrieval`) — this gives the strong Sanskrit results below.
- Model resolution order: `--model` > local `artifacts/<run_name>/` (if the notebook produced it) >
  the HF-hosted fine-tuned model > base `multilingual-e5-small` (graceful fallback if HF is unreachable).
- Retrieval only; pass an LLM to `sanskrit_retrieval.rag.answer(...)` to add grounded generation.

---

## Results (fine-tuned `multilingual-e5-small`, 1 epoch on Colab T4)

| Eval set | Metric | Base | Fine-tuned |
|---|---|---|---|
| In-domain (Itihasa test) | Recall@1 | 0.147 | **0.789** |
| In-domain | Recall@10 | 0.331 | **0.937** |
| In-domain | nDCG@10 | 0.230 | **0.864** |
| **Out-of-domain** (Bhagavad Gita, cross-domain) | Recall@1 | 0.174 | **0.718** |
| Out-of-domain | Recall@10 | 0.370 | **0.940** |

**Bonus:** `e5-base` reaches Recall@1 **0.877**; INT8 embedding quantization retains **100%** of
Recall@10 at **4× smaller** index. Devanagari tokenizer fertility measured at **3.36 tokens/word**.
Full tables, failure analysis, and the honest OOD framing are in [`report/REPORT.md`](report/REPORT.md).

## Model weights
The fine-tuned checkpoint (~471 MB) is **too large for GitHub's 100 MB file limit**, so it is hosted on
the **Hugging Face Hub**: [`AniruddhaAI/sanskrit-e5-small-retrieval`](https://huggingface.co/AniruddhaAI/sanskrit-e5-small-retrieval).
`mini_rag.py` and the notebook download it automatically. It was produced by the training notebook and
published with:
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("artifacts/e5-small-sa-en-mnrl").push_to_hub("sanskrit-e5-small-retrieval")
```

---

## Full pipeline — the Colab notebook

`notebooks/sanskrit_english_retrieval.ipynb` runs the whole thing (data → train → eval → RAG demo)
and **persists all outputs to Google Drive**.

1. Upload this folder to Google Drive.
2. Open the notebook in Colab → **Runtime → GPU (T4)**.
3. In cell 1, set `PROJECT_DIR` to the uploaded folder's path (e.g. `/content/drive/MyDrive/sanskrit-english-retrieval`).
4. (Optional) run the Hugging Face login cell to unlock the gated IN22/FLORES OOD sets — otherwise the
   notebook falls back to the ungated Bhagavad-Gita cross-domain eval.
5. Run all cells. Checkpoints land in `artifacts/`, metrics/charts/sample outputs in `results/`.

### Local tests (no GPU)
```bash
pip install pytest && python -m pytest -q     # pure-logic unit tests (normalization, pairs, metrics)
```

---

## Repo structure
```
├── mini_rag.py                 # ⚡ standalone clone-and-run retrieval demo
├── README.md · requirements.txt
├── report/REPORT.md · REPORT.pdf   # the graded report (9 sections)
├── notebooks/                  # the Colab notebook (sanskrit_english_retrieval.ipynb)
├── src/sanskrit_retrieval/     # config · normalize · data · model · train · evaluate · index · rag · rerank · bonus
├── tests/                      # pytest (pure logic, no ML deps)
├── results/                    # metrics CSVs, chart, sample outputs (generated by the notebook)
├── artifacts/                  # trained checkpoints (git-ignored — hosted on HF Hub instead)
├── 01–05 *.md                  # research → decision → technical design → roadmap → production design
└── alternatives-considered.md  # Options 1 & 3 (the tracks evaluated but not chosen)
```

## Design choices (short)
- **Model:** `multilingual-e5-small` — XLM-R lineage genuinely covers Sanskrit (CC-100), MIT, T4-friendly.
- **Data:** `rahular/itihasa` ~93K aligned verse pairs (train + in-domain test); Bhagavad Gita (demo +
  cross-domain OOD); IN22/FLORES optional gated OOD — all from the assignment's suggested list.
- **Training:** contrastive `MultipleNegativesRankingLoss` (in-batch negatives), fp16, 25% IAST
  transliteration augmentation. Full rationale + rejected alternatives in [`03`](03-technical-design.md).

## Deliverables → this repo
Colab notebook ✓ · training/inference scripts (`src/`) ✓ · README ✓ · report (MD + PDF) ✓ ·
sample outputs (`results/`) ✓ · eval scripts (`src/evaluate.py`, `tests/`) ✓ · mini-RAG (`mini_rag.py`) ✓ ·
bonus: quantization, e5-base comparison, tokenizer/fertility, cross-lingual eval ✓.

**Optional "interesting areas" (spec) — all addressed:** cross-lingual alignment · transliteration
mismatch (IAST augmentation) · chunking (verses are atomic — no splitting) · embedding L2-normalization ·
retrieval failure modes. Details in [`report/REPORT.md`](report/REPORT.md).

## License / data provenance
Model MIT · Itihasa public-domain · Bhagavad Gita public-domain source · IN22 open / FLORES CC-BY-SA
(both gated) — clean for reuse. Sanskrit sacred/classical text handled with attribution and respect.
