# Truffle Memory System Design

## 1. Problem and Constraints

Build an onboarding + chat memory backend that makes a local/open model feel deeply personal after ingesting large personal message history.

Hard constraints for this project:
- Model family budget: only 30-40B class models.
- Primary model: `qwen/qwen3-30b-a3b-instruct-2507` via OpenRouter.
- Must support effective retrieval from datasets much larger than model context (target: 5M tokens, runtime budget: <= 128K context used per turn).
- Source data: top 10 WhatsApp exports in `.txt` format.
- Must preserve raw user wording while still being searchable across multilingual/code-mixed text.

Design priority:
- Reliability and auditability over complexity.
- No fragile single-pass pipeline.
- No reliance on a bigger model for ingestion.

## 2. Key Design Decisions

1. Hierarchical memory (`raw -> episodic summaries -> profile facts`) to compress 5M into actionable context windows.
2. Dual-stream Rosetta indexing: normalized English for retrieval, raw text for final grounded responses.
3. LangGraph state graph (not ad-hoc while loops) for retrieval + recall orchestration.
4. Hybrid storage: SQLite for canonical records + ChromaDB for semantic retrieval.
5. Tiered ingestion with salience gating to control cost and latency.

## 3. Why Not a Bigger Model for Summarization

Even if a bigger model could summarize faster/better, using it for ingestion can violate the stated challenge intent (30-40B model bounded system). This design keeps **both ingestion and chat** under the same model budget.

## 4. Data and Memory Model

### 4.1 Storage Layers

- `SQLite (memory.db)` as source of truth:
  - `messages`: parsed raw messages + timestamps + sender + chat id
  - `episodes`: chunked units with salience scores
  - `summaries`: daily/weekly compressed memory with evidence ids
  - `facts`: profile facts with confidence and temporal metadata
  - `ingestion_jobs`: checkpoints/idempotency
  - `retrieval_logs`: observability/evaluation

- `ChromaDB` collections:
  - `episodic_summaries` (normalized text embedding, references episode ids)
  - `raw_episodes_index` (normalized chunk text, references raw message ids)

- `profile.json`:
  - compact runtime snapshot generated from `facts` table.

### 4.2 Fact Schema (anti-hallucination)

Each fact stores:
- `fact_text`
- `category` (identity/preference/relationship/event/goal)
- `confidence` (0-1)
- `first_seen_at`
- `last_seen_at`
- `evidence_message_ids` (required)

Facts without evidence are not promoted to profile.

## 5. Ingestion Pipeline

## 5.1 Parse and Normalize

1. Parse WhatsApp exports with multiline message reconstruction.
2. Keep original UTF-8 raw text exactly.
3. Normalize unicode quirks and message artifacts for search stream only.

## 5.2 PII and Secret Guarding

At ingestion time:
- Detect sensitive patterns (account number, IFSC, UPI, OTP, passwords, card-like patterns, Aadhaar/PAN-like patterns).
- Mark message-level sensitivity flags.
- Generate redacted search text for embedding where required.

At response time:
- Secret-bearing fields are not surfaced unless policy allows.
- If asked for passwords/OTP/card details, model refuses and states memory safety constraints.

## 5.3 Tiered Compression (Throughput Control)

For each episode chunk:
- Tier A (always): store raw + metadata + shallow extractive features.
- Tier B (selective): medium summary for moderate salience.
- Tier C (deep): richer semantic summary + fact extraction for high salience chunks.

Salience score uses:
- Recency
- User-centricity (messages by/with user)
- Repetition frequency
- Entity/event density
- Sentiment/urgency markers

Low-salience old chunks are pass-through indexed (no deep LLM compression).

## 5.4 Throughput Math and Mitigation

Naive sequential estimate:
- 5M tokens / 1K-token chunks = ~5,000 chunks
- 2s per chunk => ~10,000s (~2.7h) just first-pass summarization

Mitigations without breaking model constraints:
- Parallel workers for map stage (bounded concurrency)
- Idempotent chunk hashing (skip unchanged chunks)
- Tiered summarization (deep only for salient chunks)
- Incremental backfill by time windows
- Fast extractive fallback for backlog catch-up

Expected outcome: major reduction in wall-clock time while keeping model class fixed.

## 6. Rosetta Dual-Stream Indexing

For each chunk:
- `raw_text`: preserved exactly for grounded quoting and tone matching.
- `normalized_text`: concise semantic English representation for embedding retrieval.
- `language_tags`: detected scripts/languages.

Retrieval rank is computed on normalized text; final answer context includes raw text segments.

## 7. Runtime: LangGraph-Orchestrated Recall

Use LangGraph with explicit state and bounded loop count.

Graph nodes:
1. `load_profile`
2. `retrieve_summaries`
3. `retrieve_raw`
4. `compose_context`
5. `answer_or_request_recall`
6. `recall_retrieve` (conditional edge)
7. `finalize`

State includes:
- `user_query`
- `profile_blob`
- `context_blocks`
- `token_budget`
- `recall_count`
- `max_recall_count`
- `retrieval_trace`

Loop constraints:
- `max_recall_count = 2 or 3`
- each recall must carry a structured query intent
- no unbounded free-form recursion

## 8. Retrieval and Context Budgeting

Per turn token budget target (`<=128K` practical use):
- Profile facts: ~5K
- Episodic summaries: ~20K
- Raw evidence drill-down: ~40-60K
- Reserve for dialogue + answer generation: rest

Ranking strategy:
- Semantic score (Chroma)
- Lexical score (SQLite FTS/BM25)
- Recency decay boost
- Salience boost

Reranking returns evidence-first packs; low-confidence memories are labeled as uncertain.

## 9. Failure Modes and Risk Register

1. **Ingestion throughput + freshness lag** (missing risk from initial verdict):
   - Risk: long backfills mean stale memory in chat.
   - Mitigation: tiered ingestion, parallel workers, incremental checkpoints, freshness watermark in responses.

2. Parser drift from export format quirks:
   - Mitigation: strict tests on multiline/system messages and timestamp variants.

3. PII leakage risk:
   - Mitigation: sensitivity flags + redaction + runtime refusal policy.

4. Summary distortion:
   - Mitigation: evidence ids, confidence scoring, periodic re-grounding to raw.

5. Retrieval misses on code-mixed text:
   - Mitigation: Rosetta normalization + raw fallback + hybrid lexical retrieval.

## 10. 10-Hour Delivery Plan

1. Implement robust parser + canonical SQLite schema.
2. Implement chunker + salience scoring + idempotent ingestion.
3. Add Rosetta normalization and Chroma indexing.
4. Build LangGraph retrieval/recall graph with token budget guardrails.
5. Add safety policy for secrets and unsupported claims.
6. Run eval set (positive + adversarial + privacy prompts) and log retrieval traces.

## 11. If More Time

1. Build entity relationship graph on top of facts.
2. Add contradiction resolution over time.
3. Add temporal-aware reranker and memory decay/rehearsal.
4. Build a memory inspector UI for provenance and debugging.
5. Add automated regression suite across exports and prompt attacks.

## 12. Why This Approach

This design is intentionally conservative: it is creative enough (hierarchical + dual-stream + graph recall), but avoids brittle complexity by keeping a strict evidence chain, explicit state machine orchestration, and operational controls for ingestion throughput and privacy.
