"""Message recipients for new contacts, using a snapshot of the batch start."""

from collections.abc import Iterable, Set
from typing import Literal

MessageRule = Literal["source_to_target", "target_to_source"]


def batch_recipients(
    edges: Iterable[tuple[int, int]],
    informed: Set[int],
    rule: MessageRule = "target_to_source",
) -> set[int]:
    """Return distinct newly informed vertices; never mutate the supplied snapshot."""
    if rule not in {"source_to_target", "target_to_source"}:
        raise ValueError("rule must be 'source_to_target' or 'target_to_source'")
    snapshot = frozenset(informed)
    contacts = edges if rule == "source_to_target" else ((v, u) for u, v in edges)
    return {receiver for sender, receiver in contacts if sender in snapshot} - snapshot
