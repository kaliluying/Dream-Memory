from __future__ import annotations

from pathlib import Path
from typing import Any

DREAM_MEMORY_START = "<!-- DREAM_MEMORY_START -->"
DREAM_MEMORY_END = "<!-- DREAM_MEMORY_END -->"


def replace_marked_block(existing: str, content: str, *, heading: str = "## Dream Memory Context") -> str:
    block = f"{DREAM_MEMORY_START}\n{content.rstrip()}\n{DREAM_MEMORY_END}"
    if DREAM_MEMORY_START in existing and DREAM_MEMORY_END in existing:
        before, rest = existing.split(DREAM_MEMORY_START, 1)
        _, after = rest.split(DREAM_MEMORY_END, 1)
        return before.rstrip() + "\n\n" + block + after
    prefix = existing.rstrip()
    if prefix:
        return prefix + "\n\n" + heading + "\n\n" + block + "\n"
    return heading + "\n\n" + block + "\n"


def write_marked_file(path: Path | str, content: str, *, heading: str) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.read_text(encoding="utf-8") if output.exists() else ""
    output.write_text(replace_marked_block(existing, content, heading=heading), encoding="utf-8")
    return output


def render_all_projects_summary(memory_cards: list[dict[str, Any]]) -> str:
    active = [card for card in memory_cards if card.get("status") == "active"]
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
