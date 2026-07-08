# Production System Design — Sanskrit↔English Semantic Retrieval Service

This document is written in two layers. The **take-home scope** describes what actually ships in the graded submission: a fine-tuned `intfloat/multilingual-e5-small` (XLM-R lineage, MIT, 118M params, 384-dim) trained with SentenceTransformers `MultipleNegativesRankingLoss` on ~93K aligned verse pairs from `rahular/itihasa`, a FAISS index, offline Recall@K/MRR/nDCG evaluation, and a notebook-grade mini-RAG demo — all single-node, single-notebook, single Colab T4/L4. The **production design** describes how that same artifact becomes a hosted retrieval/RAG microservice — the parts that are *designed here but not built by the take-home*. The framing is deliberate: the take-home proves the model and the retrieval quality; this document proves I know what it takes to run it. Nothing below claims the 1–2 day submission built distributed infrastructure — where a section has no take-home footprint, it says so.

The system decomposes into five production components referenced throughout: (1) the **embedding/inference service** serving the fine-tuned e5-small, (2) the **vector index** (FAISS in the take-home → managed Qdrant/pgvector/Pinecone at scale), (3) the **query/ingest API**, (4) the **RAG orchestration layer**, and (5) the **LLM answer step**.

---

## 1. Scaling

**Take-home scope.** Single process. The model is loaded once in the notebook; the FAISS index (a flat `IndexFlatIP` over ~93K–186K vectors) lives in RAM; queries are served one at a time in a demo loop. No concurrency, no autoscaling, no sharding. This is correct for the assignment — 186K × 384 × 4 bytes ≈ 285 MB fits comfortably in T4 RAM.

**Production design.** The workload is overwhelmingly read-heavy (retrieval QPS ≫ ingest writes), and the two hot paths — *embed the query* and *search the index* — scale independently, so they are split into separate services.

| Concern | Approach |
|---|---|
| Embedding service | Stateless replicas behind a load balancer. Serve via **Text Embeddings Inference (TEI)** or ONNX Runtime; e5-small is small enough that CPU serving is viable at low/medium QPS, GPU only when batch throughput demands it. Autoscale on p95 latency + queue depth, not CPU alone. |
| Query batching | TEI/Triton **dynamic batching** — coalesce concurrent single-query requests into a micro-batch (e.g. 4–8 ms window) to amortize the forward pass. Biggest single throughput lever for embedding inference. |
| Vector index | Move off in-process FAISS to a managed vector DB (Qdrant / pgvector / Pinecone) that owns replication and sharding. **Replicate** for read QPS and HA; **shard** only when the corpus outgrows a node's RAM (classical Sanskrit corpora are small — tens of millions of chunks at most — so a single well-provisioned node + replicas covers most realistic scales). |
| Cold vs warm | Model weights and index must be memory-resident before a replica joins the LB pool. Use readiness probes gated on "model loaded + index attached + warm-up query passed." Keep a warm pool; avoid scale-to-zero on the serving path (cold-start = model load + index mmap = seconds, unacceptable for interactive search). Scale-to-zero is fine for dev only (§12). |
| Model serving | e5-small requires the `query:` / `passage:` prefix convention — enforce it in the serving layer, not the client, so all callers get correct embeddings. |

Sharding note: prefer **replication before sharding**. Sharding a vector index fragments recall (each shard returns local top-k, then you merge) and adds a fan-out/merge cost; only pay it when a single node genuinely can't hold the corpus.

---

## 2. Optimization

**Take-home scope.** `IndexFlatIP` (exact, brute-force cosine via normalized inner product). Correct choice for the assignment: at ~186K vectors it's fast, gives ground-truth recall (no ANN approximation to confound the eval), and keeps the metrics honest. No quantization, no caching.

**Production design.** Two optimization surfaces — inference latency and search latency — plus caching.

**Embedding inference latency:**
- **INT8 dynamic quantization** (ONNX Runtime) — ~2–4× CPU speedup on transformer encoders with minor quality loss; re-run the offline eval after quantizing to confirm Recall@K holds (treat it as a gated model change, §6).
- **ONNX / TEI** export removes Python overhead and enables graph-level fusion.
- **Matryoshka dimension truncation** — if the model is trained/tuned to be Matryoshka-capable, serve 256- or 128-dim vectors instead of 384 for a smaller index footprint and faster search, trading a measured slice of recall. e5-small is not natively Matryoshka; this requires MRL-aware fine-tuning, so flag it as a *candidate*, not a given.

**FAISS / ANN index tradeoffs:**

| Index | Recall | Latency | Memory | When |
|---|---|---|---|---|
| Flat (IP) | Exact (100%) | O(N) scan | 1× | Take-home; small corpora; recall-critical |
| IVF (IVF-Flat/IVF-PQ) | Tunable (nprobe) | Fast | Low (PQ compresses) | Large corpora, memory-constrained; needs `train()` |
| HNSW | Very high | Fastest queries | High (graph overhead) | Interactive prod, corpus fits RAM, high QPS |

For a Sanskrit corpus that fits in RAM, **HNSW** is the production default (best latency at high recall); **IVF-PQ** only if memory forces compression. Whatever ANN is chosen, re-measure Recall@K against the Flat baseline — ANN recall is a product decision, not a default.

**Caching:**
- **Query-embedding cache** — sacred/classical text search has a long tail of repeated canonical queries (famous verses, common terms). Cache `normalize(query) → embedding` (Redis, TTL + LRU). Skips the whole forward pass on a hit.
- **Result cache** — cache `(query_hash, k, filters) → doc IDs`; invalidate on index version bump. High hit rate for popular queries.
- **Batch vs realtime** — realtime for interactive search; batch mode (large offline embedding jobs) for corpus ingestion/re-embedding, run on cheaper preemptible compute (§12).

---

## 3. Performance metrics

**Take-home scope.** Offline retrieval metrics only, computed once on a held-out split: **Recall@{1,5,10}, MRR, nDCG@10**. These are the graded numbers and the baseline the production SLOs are anchored to. Report base model vs fine-tuned to show the lift.

**Production design.** Split into SLIs (measured) and SLOs (targets); carry the offline metrics forward as *online* quality guards.

| Metric | Type | Target (illustrative) |
|---|---|---|
| Query latency p50 / p95 / p99 | Latency SLI/SLO | p95 < 150 ms (cache miss, CPU embed + HNSW search); p99 < 400 ms |
| Throughput | Capacity | Sustained QPS per replica at target p95, published for capacity planning |
| Recall@10 / nDCG@10 (online) | Quality | Must not regress below offline baseline − ε; sampled via labeled probes + feedback (§13) |
| Index freshness | Data | Time from source-corpus update → searchable; target < 1 h for incremental, tracked per ingest |
| Cost-per-query | Cost | Blended embed + search + LLM cost; alert on drift (the LLM answer step usually dominates) |
| Availability | Reliability | 99.9% on the retrieval path |
| Cache hit rate | Efficiency | Tracked for query-embedding and result caches; informs cache sizing |

Latency budget decomposition matters: instrument **embed time**, **search time**, **rerank time**, and **LLM time** separately (§13 tracing) — otherwise you can't tell whether a p95 breach is the encoder, the index, or the LLM.

---

## 4. Request handling

**Take-home scope.** A notebook function: `search(query, k) → top-k passages`, optionally piped into a single LLM call for the RAG demo. No HTTP API, no rate limiting, no validation beyond notebook-level sanity.

**Production design.** A thin, well-contracted API in front of the five components.

**API contract:**

| Endpoint | Method | Sync/Async | Purpose |
|---|---|---|---|
| `POST /v1/query` | sync | sync | Embed query → search index → return ranked passages (+ optional generated answer) |
| `POST /v1/ingest` | sync ack / async job | async | Accept documents, chunk, embed, upsert into index |
| `GET /v1/ingest/{job_id}` | sync | — | Ingest job status |
| `GET /healthz` `GET /readyz` | sync | — | Liveness / readiness (model+index warm) |

- **Sync vs async** — retrieval is sync (interactive). Ingestion is async: return a job ID, process embedding + upsert in the background, expose status.
- **Request validation** — schema-validated at the boundary (Pydantic): query length caps, `k` bounds, language/script hint, filter allowlist. Fail fast with structured errors. Reject oversized payloads before they hit the model.
- **Rate limiting** — per-API-key token-bucket (protects the GPU/CPU embed pool and the paid LLM step, which is the real cost sink).
- **Timeouts** — per-hop deadlines (embed, search, LLM) with a total request budget; cancel downstream work when the client budget is exhausted.
- **Graceful degradation** — if vector search fails or times out, **fall back to BM25/keyword** over the same corpus so search still returns *something*; if the LLM answer step fails or times out, **return retrieved passages without the generated answer** (retrieval is the core product; generation is enhancement). Never fail the whole request because generation failed.
- **Idempotency** — ingestion must be idempotent: a stable `document_id` (content hash or source-provided key) so retries upsert rather than duplicate. Critical because re-ingest/re-embed jobs (§10) replay documents.

---

## 5. Vulnerability / AppSec

**Take-home scope.** N/A as a hardened surface — it's a notebook. The one relevant discipline that *does* apply: **verify HF model provenance** (pin `intfloat/multilingual-e5-small` by revision hash, confirm MIT license) and pin dataset revision (`rahular/itihasa`). No exposed endpoint, so no injection surface.

**Production design.** OWASP LLM Top 10 is directly relevant because of the RAG/LLM step.

| Risk | Control |
|---|---|
| Prompt injection (LLM01) | Retrieved passages are *untrusted content* — never let them alter the system prompt. Strict prompt templating, delimiter isolation of retrieved text, instruction-hierarchy enforcement, output constraints. Highest-priority risk for the RAG step. |
| Input validation / injection | Validate/normalize all query input; treat query text as data, never interpolate into shell/SQL/prompt control. |
| Embedding inversion / data exfiltration (LLM06) | Embeddings can leak source text under inversion attacks. Don't expose raw vectors via the API; return doc IDs + text, not the vector. Access-control the vector DB (§11). |
| Sensitive info disclosure | The corpus is public-domain sacred text (low secrecy), but **user queries are sensitive** — apply logging/retention limits (§7). Ensure the LLM step doesn't echo other users' data. |
| Model supply chain (LLM05) | Pin model + dataset revisions; verify checksums; build models into the artifact registry from a trusted, reproducible pipeline (§6), not pulled live from HF at runtime. |
| Dependency / CVE scanning | `pip-audit`/Snyk/Dependabot in CI; SBOM generation; block deploy on critical CVEs. |
| Secrets handling | LLM API keys, vector DB creds in a secret manager (not env files in the image); short-lived, rotated. |
| DoS via expensive queries | Bound `k`, payload size, and per-key rate (§4, §11 WAF). |

---

## 6. SDLC parameters

**Take-home scope.** A single notebook + a `requirements.txt`, ideally with a fixed seed, pinned dependency versions, and pinned model/dataset revisions so the reported metrics are reproducible. That reproducibility *is* the SDLC deliverable at take-home scale — a grader must be able to re-run and get the same numbers.

**Production design.**

| Concern | Approach |
|---|---|
| Repo structure | Split: `training/` (fine-tune pipeline), `serving/` (embedding + API services), `rag/` (orchestration), `eval/` (metric harness), `infra/` (IaC). Small focused modules. |
| Branching / CI/CD | Trunk-based with short-lived branches. CI: lint → type-check → unit/integration tests → **eval-gate**. CD promotes artifacts (model, image) through dev→staging→prod. |
| Model & data versioning | Dataset via **DVC** or pinned HF Hub revision; trained models in a **model registry** (MLflow Registry / HF Hub private) with immutable version tags. Never "latest." |
| Experiment tracking | **Weights & Biases or MLflow** — log hyperparameters, loss curves, and the full Recall@K/MRR/nDCG suite per run so model versions are comparable. |
| Reproducibility | Fixed seeds, pinned deps, containerized training, recorded commit + data revision + config per model artifact. |
| **Eval-as-a-gate** | A new fine-tuned model is **not promotable** unless its offline Recall@K/nDCG meets-or-beats the incumbent on the frozen eval set (plus latency/size budget). This is the single most important MLOps control for this system — it turns "the model got better" into an automated, non-negotiable check. |

---

## 7. Compliance

**Take-home scope.** Clean and worth stating explicitly as a **strength**: the `rahular/itihasa` corpus derives from public-domain classical texts (Rāmāyaṇa/Mahābhārata translations), and `multilingual-e5-small` is **MIT-licensed**. There is no license entanglement blocking commercial or hosted use — a genuinely favorable position that many embedding projects don't enjoy. No PII in the training data.

**Production design.**

- **Licensing** — carry the clean MIT + public-domain posture into production; document it in the artifact registry. If new corpora are ingested, run a license check as an ingest gate (some translations are copyrighted even when the source text is ancient).
- **PII** — the *corpus* is low-risk, but **user queries are personal data**. Apply GDPR-style handling: minimize what's logged, define retention (e.g. raw query logs TTL'd, aggregates kept), support deletion requests, and don't ship queries to third-party LLMs without disclosure/DPA coverage.
- **Attribution / provenance** — retain source and translation attribution per chunk and surface it in results; users of classical/sacred texts expect citable provenance.
- **Cultural/religious sensitivity** — sacred text deserves respectful handling: preserve source attribution, avoid the RAG LLM step fabricating or distorting scripture (grounding + citation, refuse-when-unsure), and be careful that generated "answers" don't present model hallucination as authoritative doctrine. This is both an ethics and a product-trust concern.

---

## 8. Disaster recovery & backup

**Take-home scope.** N/A. The notebook can be re-run from scratch; the "backup" is the source dataset + the training script. Trivial to reconstruct.

**Production design.** This system has an unusually **cheap DR story**, and that's the key insight: *the vector index is fully derivable from (source corpus + model version)*. It is a cache, not a system of record.

| Asset | Backup strategy | RTO / RPO |
|---|---|---|
| Source corpus | System of record — versioned object storage, cross-region replicated | RPO ≈ 0 (immutable, versioned) |
| Model checkpoints | Immutable artifacts in the model registry, replicated | RPO ≈ 0 |
| Vector index | **Snapshots** for fast restore **+ rebuild-from-source** as the ultimate fallback | RTO = snapshot restore (minutes) or full re-embed (hours, bounded by corpus size) |
| Query/result cache | Ephemeral — no backup needed | N/A |

Because the index is reconstructible, **RPO for the index is effectively zero** (you can never truly lose data — only lose *time* re-deriving it), and RTO is bounded and known (measure the full re-embed time on current hardware and publish it). Keep periodic index snapshots so the common-case restore is minutes, not a full rebuild. **Multi-region**: replicate source + model artifacts to a second region; the index can be re-derived there, so you don't need synchronous index replication for DR — a meaningful cost saving.

---

## 9. Support & Helpdesk framework

**Take-home scope.** N/A. There is no operational surface to support.

**Production design.**

- **Tiered support** — T1 (usage/API-key/quota issues, runbook-driven), T2 (degraded quality/latency, on-call engineer), T3 (model/index defects → ML engineering).
- **Runbooks** — for the concrete failure modes: "vector DB unreachable → confirm BM25 fallback engaged," "p95 latency breach → check embed vs search vs LLM split," "bad-recall reports → pull query, reproduce against index version, check normalization."
- **On-call & incident classification** — sev levels by user impact: Sev1 retrieval down, Sev2 degraded (LLM step down but retrieval up — degraded per §4), Sev3 quality regression on a query class.
- **Feedback loop (the ML-specific part)** — bad-retrieval reports and thumbs-down signals are not just tickets; they are **training data**. Route flagged queries into a review queue → confirmed bad results become **hard negatives** for the next fine-tuning round (§13 flywheel). This closes support into model improvement, which is the whole point of owning the model.

---

## 10. System maintenance

**Take-home scope.** N/A — a one-shot artifact.

**Production design.**

| Task | Cadence / trigger |
|---|---|
| Model retraining | Periodic (e.g. quarterly) or triggered by accumulated hard negatives / measured quality drift. Gated by §6 eval-gate before promotion. |
| Index re-embedding | **Mandatory on any model upgrade** — embeddings from model v(n) and v(n+1) are not comparable, so the entire index must be re-embedded with the new model. Do it as a background batch job (§12 preemptible compute) into a **new index version**, then atomically switch the alias — zero-downtime cutover, instant rollback. |
| Dependency updates | Regular patching; CVE-driven out-of-band (§5). |
| Data drift monitoring | Track **query-distribution drift** (are users asking things unlike the training distribution? e.g. new scripts, transliteration styles, out-of-corpus topics) and recall drift on labeled probes. Drift → retrain trigger. |
| Deprecation | Version endpoints and index/model versions; announce and sunset old model versions with a migration window. |

The re-embed-on-upgrade + alias-swap pattern is the maintenance backbone: model changes and index changes are the same operation, executed as blue/green on the index.

---

## 11. Network & security requirements

**Take-home scope.** N/A — local notebook, no network surface. (Provenance pinning from §5 is the only applicable item.)

**Production design.**

| Layer | Control |
|---|---|
| Transport | **TLS** everywhere; HSTS at the edge. |
| Vector DB isolation | Vector DB in a **private VPC subnet**, no public IP; reachable only from the query/ingest services via internal networking + security groups. It holds the derived corpus and must never be internet-exposed (embedding-inversion risk, §5). |
| AuthN / AuthZ | API keys for service clients; **OAuth2/OIDC** for user-facing access; per-key scopes (query vs ingest are different privileges). |
| Network policies | Default-deny between namespaces/services; explicit allow only for the required embed→index→LLM path. |
| Egress control | The **LLM answer step is the only egress to a third party** — pin it to an allowlisted endpoint, route through an egress proxy, log all outbound. Prevents both exfiltration and surprise dependencies. |
| Edge protection | **WAF + DDoS** protection at the edge; bot/abuse rules; rate limiting (§4) as the app-layer backstop. |
| Secrets | Managed secret store, rotation, no secrets in images (§5). |

---

## 12. Resource management: efficiency through automation

**Take-home scope.** Free single Colab T4/L4. The entire compute budget is "one notebook session." That's the point — the design must prove cost-awareness *precisely because* the take-home is nearly free.

**Production design.**

| Lever | Decision |
|---|---|
| **IaC** | Everything in **Terraform** — services, vector DB, network, autoscaling policies, secrets wiring. No click-ops; environments reproducible. |
| **GPU vs CPU serving** | e5-small is small — **CPU serving (ONNX/TEI, INT8) is viable and cheaper at low/medium QPS**. Reserve GPU for the batch re-embedding jobs and only add GPU serving replicas when sustained QPS makes batched GPU throughput cheaper per query than CPU. Default to CPU; justify GPU with numbers. |
| **Spot/preemptible for batch** | Corpus (re-)embedding is idempotent, restartable, and non-interactive → run it on **spot/preemptible GPUs** with checkpointing. Big cost win on the heaviest workload. |
| **Autoscaling** | Scale the embedding service on p95 latency + queue depth; scale index replicas on read QPS. |
| **Scale-to-zero (dev only)** | Dev/staging serving scales to zero off-hours; **never** on the prod retrieval path (cold model load is user-visible, §1). |
| **Cost guardrails** | Budgets + alerts, with special attention to the **LLM answer step** (per-token cost, usually the dominant line item) and GPU hours. Cache hits (§2) directly cut both LLM and embed spend. |

---

## 13. Other necessary parameters

Items critical to *this* system that the 12 standard sections underweight:

- **Observability / tracing** — **OpenTelemetry** distributed traces across embed → search → rerank → LLM, with per-hop latency spans (feeds §3's latency-budget decomposition). Structured logs (query hash, index version, model version, k, cache hit/miss) and metrics dashboards. Without per-hop tracing you cannot debug a p95 breach in a multi-stage RAG pipeline.
- **Ingestion / chunking pipeline at scale** — verses/ślokas are naturally short and self-contained, which is convenient, but production ingest still needs a defined chunking policy (verse-level vs passage-level, overlap, metadata attachment: source, book, canto, translation attribution). Chunking choices directly move Recall@K, so they are versioned alongside the model and re-run on re-embed (§10).
- **Multilingual / transliteration handling in prod** — this is Sanskrit-specific and easy to miss: users query in **Devanāgarī, IAST, Harvard-Kyoto, or romanized ad-hoc** forms. Normalize scripts at query time (Devanāgarī↔IAST/transliteration normalization) *before* embedding, and index the corpus consistently, so "the same verse" retrieves regardless of input script. Mismatched normalization between ingest and query is a silent recall killer. Expose a script/language hint in the API (§4) to disambiguate.
- **Evaluation in production** — the offline metrics (§3) are a snapshot; production truth comes from **online A/B testing** of retrieval variants (model versions, ANN params, chunking) measured on **click/dwell/feedback signals**, plus a small continuously-labeled probe set to detect live recall regression.
- **Data flywheel** — the loop that makes owning the model worthwhile: user queries + feedback (thumbs, clicks, T1 bad-retrieval reports from §9) → hard-negative mining → next fine-tune → eval-gate (§6) → re-embed + alias swap (§10). Each turn of the wheel improves retrieval on the *actual* query distribution, not just the static Itihasa split. This is the strategic reason to fine-tune a model you control rather than call a closed embedding API.
