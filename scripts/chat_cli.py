from __future__ import annotations

import logging
from queue import Empty, Queue
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from runtime.chat import ChatSession

app = typer.Typer(help="Interactive chat CLI with LangGraph recall loop.")
console = Console()


def _configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )
    for noisy_name in ("httpx", "httpcore", "openai", "openai._base_client"):
        noisy = logging.getLogger(noisy_name)
        noisy.setLevel(logging.WARNING)
        noisy.propagate = False


def _memory_snapshot() -> dict[str, int] | None:
    settings = get_settings()
    if not settings.sqlite_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(settings.sqlite_path), timeout=2)
        conn.execute("PRAGMA busy_timeout=2000;")
        cur = conn.cursor()
        snap = {
            "messages": int(cur.execute("SELECT COUNT(*) FROM messages").fetchone()[0]),
            "episodes": int(cur.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]),
            "episodic": int(
                cur.execute("SELECT COUNT(*) FROM summaries WHERE summary_type='episodic'").fetchone()[0]
            ),
            "weekly": int(
                cur.execute("SELECT COUNT(*) FROM summaries WHERE summary_type='weekly'").fetchone()[0]
            ),
            "facts": int(cur.execute("SELECT COUNT(*) FROM facts").fetchone()[0]),
            "running_jobs": int(
                cur.execute("SELECT COUNT(*) FROM ingestion_jobs WHERE status='running'").fetchone()[0]
            ),
        }
        conn.close()
        return snap
    except sqlite3.OperationalError:
        return None


def _print_warmup_line() -> None:
    snap = _memory_snapshot()
    if snap is None:
        console.print("[dim]I'm learning your memory in the background...[/dim]")
        return
    console.print(
        "[dim]Memory warmup: "
        f"messages={snap['messages']} episodes={snap['episodes']} "
        f"summaries={snap['episodic']} weekly={snap['weekly']} facts={snap['facts']} "
        f"running_jobs={snap['running_jobs']}[/dim]"
    )


def _reset_memory_store() -> None:
    settings = get_settings()
    settings.memory_dir.mkdir(parents=True, exist_ok=True)

    removed: list[str] = []
    if settings.sqlite_path.exists():
        settings.sqlite_path.unlink()
        removed.append(str(settings.sqlite_path))
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
        removed.append(str(settings.chroma_dir))
    if settings.profile_path.exists():
        settings.profile_path.unlink()
        removed.append(str(settings.profile_path))

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    if removed:
        console.print("[dim]Fresh memory reset completed.[/dim]")
    else:
        console.print("[dim]Fresh memory reset skipped (nothing to delete).[/dim]")


def _render_benchmark(metrics: dict[str, Any]) -> None:
    turn_prompt = int(metrics.get("turn_prompt_tokens_est", metrics.get("prompt_tokens_est", 0)))
    context_window = max(1, int(metrics.get("context_window_tokens", metrics.get("prompt_budget_tokens", 1))))
    turn_pct = float(metrics.get("turn_prompt_pct", metrics.get("prompt_budget_pct", 0.0)))
    session_retained = int(metrics.get("session_retained_tokens_est", metrics.get("history_tokens_est", 0)))
    session_pct = float(
        metrics.get(
            "session_retained_pct",
            round((session_retained * 100.0) / context_window, 2),
        )
    )
    cumulative_prompt = int(metrics.get("session_cumulative_prompt_tokens_est", turn_prompt))
    turn_count = max(1, int(metrics.get("session_turn_count", 1)))
    working_memory = int(metrics.get("working_memory_tokens_est", 0))
    prompt_history = int(metrics.get("history_tokens_est", 0))
    profile_tokens = int(metrics.get("profile_tokens_est", 0))
    contexts = int(metrics.get("retrieved_context_count", 0))
    completion = int(metrics.get("completion_tokens_est", 0))
    recall_count = int(metrics.get("recall_count", 0))
    confidence_label = str(metrics.get("confidence_label", "")).strip() or "unknown"
    stream_enabled = bool(metrics.get("streaming_enabled", False))

    line = (
        f"Turn prompt {turn_prompt:,}/{context_window:,} ({turn_pct:.2f}%) | "
        f"Session retained {session_retained:,}/{context_window:,} ({session_pct:.2f}%)\n"
        f"Working memory {working_memory:,} | Prompt history {prompt_history:,} | "
        f"Profile {profile_tokens:,} | Blocks {contexts} | Completion ~{completion:,}\n"
        f"Cumulative sent ~{cumulative_prompt:,} across {turn_count} turns | "
        f"Recall {recall_count} | Confidence {confidence_label} | "
        f"Streaming {str(stream_enabled).lower()}"
    )
    console.print(Panel(line, title="Benchmark", border_style="cyan"))


def _run_chat_loop(
    session: ChatSession,
    show_trace: bool,
    show_status: bool,
    show_metrics: bool,
    stream: bool,
) -> None:
    console.print("Type `exit` or `quit` to stop.")

    turn = 0
    while True:
        if show_status and turn % 2 == 0:
            _print_warmup_line()

        try:
            query = console.input("\n[bold cyan]You > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting chat.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            console.print("Exiting chat.")
            break

        if stream:
            console.print("[bold green]OS1 > [/bold green]", end="")
            answer, trace, metrics = _ask_with_fillers(session, query)
            console.print("")
        else:
            answer, trace, metrics = session.ask(query, stream=False, on_token=None)
            console.print(Panel(answer, title="OS1", border_style="green"))
        turn += 1

        if show_metrics:
            _render_benchmark(metrics)

        if show_trace:
            console.print("[dim]Trace:[/dim]")
            for item in trace[:20]:
                console.print(f"[dim]- {item}[/dim]")


@app.command()
def run(
    show_trace: bool = typer.Option(False, help="Show retrieval trace metadata after each answer."),
    show_status: bool = typer.Option(False, help="Show memory warmup status while chatting."),
    show_metrics: bool = typer.Option(True, help="Show live token/context benchmark after each answer."),
    stream: bool = typer.Option(True, help="Stream assistant responses token-by-token."),
    log_level: str = typer.Option("WARNING", help="Logging level (DEBUG, INFO, WARNING)."),
) -> None:
    _configure_logging(log_level)
    session = ChatSession()
    _run_chat_loop(
        session=session,
        show_trace=show_trace,
        show_status=show_status,
        show_metrics=show_metrics,
        stream=stream,
    )


@app.command()
def demo(
    show_trace: bool = typer.Option(False, help="Show sanitized trace metadata after each answer."),
    show_metrics: bool = typer.Option(True, help="Show live token/context benchmark after each answer."),
    stream: bool = typer.Option(True, help="Stream assistant responses token-by-token."),
    log_level: str = typer.Option("WARNING", help="Logging level (DEBUG, INFO, WARNING)."),
) -> None:
    _configure_logging(log_level)
    _reset_memory_store()
    console.print(Panel("Hi. I'm OS1.\nI'm learning your memory now in the background.", title="OS1"))
    console.print("[dim]You can ask lightweight questions immediately.[/dim]")
    session = ChatSession()
    _run_chat_loop(
        session=session,
        show_trace=show_trace,
        show_status=True,
        show_metrics=show_metrics,
        stream=stream,
    )


def _ask_with_fillers(session: ChatSession, query: str) -> tuple[str, list[dict[str, object]], dict[str, Any]]:
    token_queue: Queue[str] = Queue()
    result_holder: dict[str, Any] = {}
    error_holder: dict[str, Exception] = {}

    def _on_token(token: str) -> None:
        token_queue.put(str(token))

    def _worker() -> None:
        try:
            answer, trace, metrics = session.ask(query, stream=True, on_token=_on_token)
            result_holder["value"] = (answer, trace, metrics)
        except Exception as exc:
            error_holder["exc"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    started = time.monotonic()
    last_fill = started
    fill_idx = 0
    seen_output = False

    while worker.is_alive() or not token_queue.empty():
        try:
            piece = token_queue.get(timeout=0.2)
            if piece:
                seen_output = True
                console.print(piece, end="", markup=False, highlight=False)
            continue
        except Empty:
            pass

        now = time.monotonic()
        if not seen_output and now - started >= 1.8 and now - last_fill >= 3.0:
            filler = _FILLER_LINES[fill_idx % len(_FILLER_LINES)]
            fill_idx += 1
            last_fill = now
            console.print(f"\n[dim]{filler}[/dim]")
            console.print("[bold green]OS1 > [/bold green]", end="")

    worker.join(timeout=1.0)

    if "exc" in error_holder:
        raise error_holder["exc"]
    value = result_holder.get("value")
    if not value:
        raise RuntimeError("Streaming response failed before result was produced.")
    return value


_FILLER_LINES = (
    "I'm thinking through your memory...",
    "I'm still with you. Pulling the right context...",
    "I'm reflecting on what I know so far...",
)


if __name__ == "__main__":
    app()
