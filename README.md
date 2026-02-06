# Truffle Memory

Personal memory backend with hierarchical ingestion, Chroma retrieval, and LangGraph recall loop.

## Quickstart

1. Copy `.env.example` to `.env` and fill `OPENROUTER_API_KEY`.
2. Put WhatsApp `.txt` exports in `data/whatsapp/`.
3. Create env and install dependencies with `uv`.
4. Run fast bootstrap ingestion (no LLM calls):
   `python scripts/ingest.py run`
5. Optional deep enrichment in background:
   `python scripts/ingest.py enrich --drain`
6. Optional weekly consolidation in background:
   `python scripts/ingest.py consolidate --drain`
7. Start chat:
   `python scripts/chat_cli.py`
8. One-command demo mode (fresh DB + fresh Chroma + autonomous background ingestion/steward):
   `python scripts/chat_cli.py demo`
9. `run` and `demo` support streaming + live benchmark box:
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
