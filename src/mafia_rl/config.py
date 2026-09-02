"""Versioned rules configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Rules which may vary between experiments.

    Spaces and checkpoints should be associated with ``schema_version`` and a
    concrete instance of this configuration.
    """

    mafia_count: int = 3
    citizen_count: int = 5
    doctor_count: int = 1
    detective_count: int = 1
    max_days: int = 10
    schema_version: str = "mafia-public-actions-v2"

    @property
    def player_count(self) -> int:
        return (
            self.mafia_count
            + self.citizen_count
            + self.doctor_count
            + self.detective_count
        )

    @property
    def non_mafia_count(self) -> int:
        return self.player_count - self.mafia_count

    def validate(self) -> None:
        counts = (
            self.mafia_count,
            self.citizen_count,
            self.doctor_count,
            self.detective_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("role counts cannot be negative")
        if self.mafia_count < 1:
            raise ValueError("at least one Mafia player is required")
        if self.non_mafia_count < 1:
            raise ValueError("at least one non-Mafia player is required")
        if self.mafia_count >= self.non_mafia_count:
            raise ValueError("the initial roster must not start at Mafia parity")
        if self.player_count < 3:
            raise ValueError("at least three players are required")
        if self.max_days < 1:
            raise ValueError("max_days must be positive")
