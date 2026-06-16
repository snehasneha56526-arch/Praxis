"""
praxis_env.memory - Agent-controlled working memory.

# Grounded in arxiv AgeMem (S28) + arxiv 2601.07190 Context Bloat (S29)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PraxisMemory:
    """Pure, deterministic memory state machine scoped to one episode."""

    saved_findings: dict[str, str] = field(default_factory=dict)
    CONTEXT_CUTOFF_STEP: int = 30

    def save_finding(self, key: str, value: str) -> str:
        """Persist or overwrite a finding and return a stable acknowledgement."""
        self.saved_findings[str(key)] = str(value)
        return f"Saved: {key}"

    def recall_memory(self, key: str | None = None) -> str:
        """Recall one finding by key, or all saved findings if key is omitted."""
        if key is not None:
            if key in self.saved_findings:
                return self.saved_findings[key]
            return f"No saved finding for key: {key}"

        if not self.saved_findings:
            return "No findings saved."

        lines = [
            f"{saved_key}: {self.saved_findings[saved_key]}"
            for saved_key in sorted(self.saved_findings)
        ]
        return "\n".join(lines)

    def get_observation_context(self, full_log: list[str], step: int) -> str:
        """
        Return visible investigation context based on cutoff state.

        Before cutoff: return the last 10 entries of the raw log.
        After cutoff: return the deterministic memory-only banner.
        """
        if step < self.CONTEXT_CUTOFF_STEP:
            return "\n".join(full_log[-10:]) if full_log else ""

        return (
            f"[CONTEXT LIMIT REACHED — Step {step}/{self.CONTEXT_CUTOFF_STEP}]\n"
            "Full investigation log is no longer available.\n"
            f"Your saved findings:\n{self.recall_memory()}\n\n"
            "Use recall_memory to retrieve specific findings.\n"
            "Use save_finding to persist new evidence."
        )

    def is_active(self, step: int) -> bool:
        """Return whether memory cutoff mode is active at this step."""
        return step >= self.CONTEXT_CUTOFF_STEP

    def reset(self) -> None:
        """Clear all saved findings for a new episode."""
        self.saved_findings.clear()
