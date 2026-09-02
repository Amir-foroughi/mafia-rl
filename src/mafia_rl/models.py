"""Domain models shared by the engine, agents, and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Union


PlayerId = int


class Alignment(str, Enum):
    MAFIA = "mafia"
    CITIZEN = "citizen"


class Role(str, Enum):
    MAFIA = "mafia"
    CITIZEN = "citizen"
    DOCTOR = "doctor"
    DETECTIVE = "detective"

    @property
    def alignment(self) -> Alignment:
        return Alignment.MAFIA if self is Role.MAFIA else Alignment.CITIZEN


class Phase(str, Enum):
    DAY_TARGET = "day_target"
    NOMINATION = "nomination"
    RUNOFF = "runoff"
    DEFENCE_VOTE = "defence_vote"
    NIGHT = "night"
    TERMINAL = "terminal"


class DecisionKind(str, Enum):
    DAY_TARGET = "day_target"
    NOMINATION = "nomination"
    RUNOFF = "runoff"
    DEFENCE_VOTE = "defence_vote"
    MAFIA_KILL = "mafia_kill"
    PROTECT = "protect"
    INVESTIGATE = "investigate"


class Visibility(str, Enum):
    PUBLIC = "public"
    MAFIA = "mafia"
    PRIVATE = "private"
    ENGINE = "engine"


class DeathPhase(str, Enum):
    DAY = "day"
    NIGHT = "night"


@dataclass(frozen=True)
class DayTarget:
    targets: tuple[PlayerId, ...]


@dataclass(frozen=True)
class NominationVote:
    targets: tuple[PlayerId, ...]


@dataclass(frozen=True)
class RunoffVote:
    targets: tuple[PlayerId, ...]


@dataclass(frozen=True)
class DefenceVote:
    target: PlayerId


@dataclass(frozen=True)
class MafiaKillVote:
    target: PlayerId


@dataclass(frozen=True)
class Protect:
    target: PlayerId


@dataclass(frozen=True)
class Investigate:
    target: PlayerId


Action = Union[
    DayTarget,
    NominationVote,
    RunoffVote,
    DefenceVote,
    MafiaKillVote,
    Protect,
    Investigate,
]


@dataclass
class PlayerState:
    player_id: PlayerId
    role: Role
    alive: bool = True

    @property
    def alignment(self) -> Alignment:
        return self.role.alignment


@dataclass(frozen=True)
class Elimination:
    player_id: PlayerId
    day: int
    phase: DeathPhase


@dataclass(frozen=True)
class Event:
    sequence: int
    event_type: str
    day: int
    visibility: Visibility
    actor: PlayerId | None = None
    recipients: tuple[PlayerId, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class GameState:
    day: int
    phase: Phase
    players: dict[PlayerId, PlayerState]
    events: list[Event] = field(default_factory=list)
    eliminations: list[Elimination] = field(default_factory=list)
    day_targets: dict[PlayerId, tuple[PlayerId, ...]] = field(default_factory=dict)
    nomination_votes: dict[PlayerId, tuple[PlayerId, ...]] = field(default_factory=dict)
    original_nomination_totals: dict[PlayerId, int] = field(default_factory=dict)
    locked_defendants: tuple[PlayerId, ...] = ()
    runoff_candidates: tuple[PlayerId, ...] = ()
    runoff_open_seats: int = 0
    defendants: tuple[PlayerId, ...] = ()
    announced_eliminated: int = 0
    announced_mafia: int = 0
    revealed_alignments: dict[PlayerId, Alignment] = field(default_factory=dict)
    winner: Alignment | None = None
    terminated: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class DecisionRequest:
    player_id: PlayerId
    kind: DecisionKind
    legal_actions: tuple[Action, ...]


@dataclass(frozen=True)
class Observation:
    player_id: PlayerId
    own_role: Role
    own_alignment: Alignment
    day: int
    phase: Phase
    alive: Mapping[PlayerId, bool]
    public_living_mafia: int
    announced_eliminated: int
    announced_mafia: int
    revealed_alignments: Mapping[PlayerId, Alignment]
    mafia_teammates: tuple[PlayerId, ...]
    investigation_results: Mapping[PlayerId, Alignment]
    events: tuple[Event, ...]


@dataclass(frozen=True)
class Transition:
    phase_before: Phase
    phase_after: Phase
    events: tuple[Event, ...]
    rewards: Mapping[PlayerId, float]
    terminated: bool
    truncated: bool
    winner: Alignment | None


class InvalidActionError(ValueError):
    """Raised when an actor or action is not legal for the current phase."""
