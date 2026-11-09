"""Shared result types.

The central idea is :class:`Contribution`. Every physical effect that could enter
the budget returns one, including effects that were judged not to apply. The
reason text is shown in the app and copied into the report, so the budget explains
itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Contribution:
    """One term in a budget, and why it was or was not used."""

    name: str
    value: float
    unit: str
    applicable: bool
    reason: str
    reference: str = ""
    validity_note: str = ""

    @property
    def value_if_applied(self) -> float:
        """The value to add to the budget: zero if the effect was rejected."""
        return self.value if self.applicable else 0.0


def contributions_table(items: list[Contribution]) -> list[dict]:
    """Rows for display in the app or the report."""
    rows = []
    for c in items:
        rows.append(
            {
                "Effect": c.name,
                "Value": f"{c.value:.2f} {c.unit}",
                "Used": "Yes" if c.applicable else "No",
                "Why": c.reason,
                "Reference": c.reference,
                "Validity note": c.validity_note,
            }
        )
    return rows
