from __future__ import annotations

from pathlib import Path
from typing import Any

from .memory_dreaming import contains_sensitive_memory_text

DREAM_MEMORY_START = "<!-- DREAM_MEMORY_START -->"
DREAM_MEMORY_END = "<!-- DREAM_MEMORY_END -->"


def replace_marked_block(existing: str, content: str, *, heading: str = "## Dream Memory Context") -> str:
    block = f"{DREAM_MEMORY_START}\n{content.rstrip()}\n{DREAM_MEMORY_END}"
    start_index = existing.find(DREAM_MEMORY_START)
    end_index = existing.find(DREAM_MEMORY_END, start_index + len(DREAM_MEMORY_START)) if start_index != -1 else -1
    if start_index != -1 and end_index != -1:
        before = existing[:start_index]
        after = existing[end_index + len(DREAM_MEMORY_END):]
        return before.rstrip() + "\n\n" + block + after
    prefix = existing.rstrip()
    if prefix:
        return prefix + "\n\n" + heading + "\n\n" + block + "\n"
    return heading + "\n\n" + block + "\n"


def validate_marked_file_targets(paths: list[Path | str]) -> None:
    for path in paths:
        output = Path(path).expanduser()
        if output.exists() and not output.is_file():
            raise ValueError(f"export target path is not writable: {output}")
        if output.parent.exists() and not output.parent.is_dir():
            raise ValueError(f"export target path is not writable: {output}")


def write_text_file_atomic(path: Path | str, content: str, *, error_label: str = "output") -> Path:
    output = Path(path).expanduser()
    if output.exists() and not output.is_file():
        raise ValueError(f"{error_label} path is not writable: {output}")
    if output.parent.exists() and not output.parent.is_dir():
        raise ValueError(f"{error_label} path is not writable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(output)
    except OSError as exc:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise ValueError(f"{error_label} path is not writable: {output}") from exc
    return output


def write_marked_file(path: Path | str, content: str, *, heading: str) -> Path:
    output = Path(path).expanduser()
    validate_marked_file_targets([output])
    existing = output.read_text(encoding="utf-8") if output.exists() else ""
    return write_text_file_atomic(
        output,
        replace_marked_block(existing, content, heading=heading),
        error_label="export target",
    )


def render_all_projects_summary(memory_cards: list[dict[str, Any]]) -> str:
    active = [card for card in memory_cards if card.get("status") == "active" and not contains_sensitive_memory_text(card)]
    user_global = [card for card in active if card.get("scope") in {"user", "global"}]
    projects: dict[str, list[dict[str, Any]]] = {}
    for card in active:
        if card.get("scope") == "project":
            projects.setdefault(str(card.get("project") or "unknown"), []).append(card)

    lines = ["# Dream Memory Project Summary", ""]
    lines.append("## User / Global")
    if user_global:
        for card in sorted(user_global, key=lambda item: (str(item.get("scope")), str(item.get("summary")))):
            lines.append(f"- **{card.get('scope')} / {card.get('memory_type')}**: {card.get('summary')}")
    else:
        lines.append("- No active user/global memory.")

    for project in sorted(projects):
        lines.extend(["", f"## {project}"])
        for card in sorted(projects[project], key=lambda item: (str(item.get("memory_type")), str(item.get("summary")))):
            lines.append(f"- **{card.get('memory_type')}**: {card.get('summary')}")
    return "\n".join(lines) + "\n"
