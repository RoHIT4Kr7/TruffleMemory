# Truffle Memory

Personal memory backend with hierarchical ingestion, Chroma retrieval, and LangGraph recall loop.

## Quickstart

1. Add your WhatsApp exports first: create `data/whatsapp/` (inside the memory data area) and place all chat `.txt` files there.
2. Create a virtual environment and install dependencies:
   `uv venv`
   `.\.venv\Scripts\activate`
   `uv add -r requirements.txt`
3. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.
4. Start chat with streaming + benchmark + warmup status:
   `python scripts/chat_cli.py demo --stream --show-metrics --show-status`

Optional manual ingestion commands:

1. Fast bootstrap ingestion:
   `python scripts/ingest.py run`
2. Deep enrichment:
   `python scripts/ingest.py enrich --drain`
3. Weekly consolidation:
   `python scripts/ingest.py consolidate --drain`
4. One-command demo mode (fresh DB + fresh Chroma + autonomous background ingestion/steward):
   `python scripts/chat_cli.py demo --stream --show-metrics`
   In streaming mode, recall loops are capped for responsiveness during live interaction.

## Autonomous Memory Steward

During chat, a runtime steward can schedule background memory maintenance without blocking responses:

- Fast ingest bootstrap (`ingest_all(force=False)`) once per chat session.
- Deep enrichment batches (`enrich_deep_episodes`) when retrieval is weak or recall loops trigger.
- Deep enrichment batches when answer confidence is low (model signal + retrieval heuristics).
- Weekly consolidation batches (`consolidate_weekly_summaries`) when pending weekly groups exist.

User-facing trust UX:

- Low-confidence answers can include lightweight reflection language (for example, "I might need to reflect on this. I'll come back stronger.").
- Internal job names and numeric confidence scores are not exposed in user-visible trace output.

Autonomy mode:

- Opening chat starts session-scoped stewardship automatically (fast ingest kickoff + bounded maintenance passes).
- No permanent daemon is required; work is scoped to active chat sessions.
- `python scripts/chat_cli.py demo` resets persisted memory first, then starts this autonomy flow in one command.

Optional env controls:

- `STEWARD_AUTO_BOOTSTRAP=true|false`
- `STEWARD_COOLDOWN_SECONDS=45`
- `STEWARD_ENRICH_BATCH=8`
- `STEWARD_CONSOLIDATE_BATCH=3`
- `STEWARD_MIN_CONTEXTS=5`
- `STEWARD_SCORE_THRESHOLD=0.58`
- `STEWARD_LOW_CONFIDENCE_THRESHOLD=0.45`
