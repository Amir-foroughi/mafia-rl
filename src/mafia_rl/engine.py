"""Deterministic, headless Mafia game engine."""

from __future__ import annotations

import itertools
import random
from collections import Counter
from collections.abc import Mapping

from .config import GameConfig
from .models import (
    Action,
    Alignment,
    DayTarget,
    DeathPhase,
    DecisionKind,
    DecisionRequest,
    DefenceVote,
    Elimination,
    Event,
    GameState,
    Investigate,
    InvalidActionError,
    MafiaKillVote,
    NominationVote,
    Observation,
    Phase,
    PlayerId,
    PlayerState,
    Protect,
    Role,
    RunoffVote,
    Transition,
    Visibility,
)


class GameEngine:
    """Resolve one Mafia episode from submitted joint actions."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self.config.validate()
        self._rng = random.Random()
        self._state: GameState | None = None

    @property
    def state(self) -> GameState:
        if self._state is None:
            raise RuntimeError("reset() must be called before accessing state")
        return self._state

    def reset(self, seed: int | None = None) -> GameState:
        self._rng = random.Random(seed)
        roles = (
            [Role.MAFIA] * self.config.mafia_count
            + [Role.DOCTOR] * self.config.doctor_count
            + [Role.DETECTIVE] * self.config.detective_count
            + [Role.CITIZEN] * self.config.citizen_count
        )
        self._rng.shuffle(roles)
        players = {
            player_id: PlayerState(player_id, role)
            for player_id, role in enumerate(roles)
        }
        self._state = GameState(day=1, phase=Phase.DAY_TARGET, players=players)
        self._emit(
            "game_started",
            Visibility.PUBLIC,
            payload={"schema_version": self.config.schema_version},
        )
        self._emit("episode_seed", Visibility.ENGINE, payload={"seed": seed})
        mafia = self._mafia_ids()
        self._emit(
            "night_zero_mafia_reveal",
            Visibility.MAFIA,
            recipients=mafia,
            payload={"mafia": mafia},
        )
        for player in players.values():
            self._emit(
                "role_assigned",
                Visibility.PRIVATE,
                recipients=(player.player_id,),
                payload={"role": player.role.value},
            )
        self._announce_day()
        return self.state

    def current_requests(self) -> dict[PlayerId, DecisionRequest]:
        if self.state.terminated or self.state.truncated:
            return {}
        requests: dict[PlayerId, DecisionRequest] = {}
        for player_id, kind in self._decision_actors().items():
            requests[player_id] = DecisionRequest(
                player_id=player_id,
                kind=kind,
                legal_actions=self.legal_actions(player_id),
            )
        return requests

    def legal_actions(self, player_id: PlayerId) -> tuple[Action, ...]:
        actor_kinds = self._decision_actors()
        if player_id not in actor_kinds:
            return ()
        kind = actor_kinds[player_id]
        alive = self._alive_ids()

        if kind is DecisionKind.DAY_TARGET:
            candidates = tuple(pid for pid in alive if pid != player_id)
            maximum = min(self.public_living_mafia, len(candidates))
            return tuple(
                DayTarget(combo)
                for size in range(1, maximum + 1)
                for combo in itertools.combinations(candidates, size)
            )

        if kind is DecisionKind.NOMINATION:
            linked = set(self._linked_players(player_id, alive))
            actions: list[Action] = []
            for combo in itertools.combinations(alive, 2):
                required = min(2, len(linked))
                if len(linked.intersection(combo)) >= required:
                    actions.append(NominationVote(combo))
            return tuple(actions)

        if kind is DecisionKind.RUNOFF:
            candidates = self.state.runoff_candidates
            open_seats = self.state.runoff_open_seats
            linked = set(self._linked_players(player_id, candidates))
            actions = []
            for combo in itertools.combinations(candidates, open_seats):
                required = min(open_seats, len(linked))
                if len(linked.intersection(combo)) >= required:
                    actions.append(RunoffVote(combo))
            return tuple(actions)

        if kind is DecisionKind.DEFENCE_VOTE:
            return tuple(DefenceVote(pid) for pid in self.state.defendants)

        if kind is DecisionKind.MAFIA_KILL:
            return tuple(
                MafiaKillVote(pid)
                for pid in alive
                if self.state.players[pid].alignment is Alignment.CITIZEN
            )

        if kind is DecisionKind.PROTECT:
            return tuple(Protect(pid) for pid in alive)

        if kind is DecisionKind.INVESTIGATE:
            return tuple(Investigate(pid) for pid in alive if pid != player_id)

        raise AssertionError(f"unhandled decision kind: {kind}")

    @property
    def public_living_mafia(self) -> int:
        return self.config.mafia_count - self.state.announced_mafia

    def observe(self, player_id: PlayerId) -> Observation:
        player = self.state.players[player_id]
        visible_events = tuple(
            event
            for event in self.state.events
            if self._event_visible_to(event, player_id)
        )
        mafia_teammates = (
            tuple(pid for pid in self._mafia_ids() if pid != player_id)
            if player.alignment is Alignment.MAFIA
            else ()
        )
        investigation_results: dict[PlayerId, Alignment] = {}
        for event in visible_events:
            if event.event_type == "investigation_result" and event.actor == player_id:
                investigation_results[int(event.payload["target"])] = Alignment(
                    event.payload["alignment"]
                )
        return Observation(
            player_id=player_id,
            own_role=player.role,
            own_alignment=player.alignment,
            day=self.state.day,
            phase=self.state.phase,
            alive={pid: item.alive for pid, item in self.state.players.items()},
            public_living_mafia=self.public_living_mafia,
            announced_eliminated=self.state.announced_eliminated,
            announced_mafia=self.state.announced_mafia,
            revealed_alignments=dict(self.state.revealed_alignments),
            mafia_teammates=mafia_teammates,
            investigation_results=investigation_results,
            events=visible_events,
        )

    def step(self, actions: Mapping[PlayerId, Action]) -> Transition:
        if self.state.terminated or self.state.truncated:
            raise RuntimeError("cannot step a completed game")
        phase_before = self.state.phase
        event_start = len(self.state.events)
        requests = self.current_requests()
        if set(actions) != set(requests):
            missing = sorted(set(requests) - set(actions))
            extra = sorted(set(actions) - set(requests))
            raise InvalidActionError(f"joint action actors mismatch; missing={missing}, extra={extra}")
        for player_id, action in actions.items():
            if action not in requests[player_id].legal_actions:
                raise InvalidActionError(
                    f"illegal action for player {player_id}: {action!r}"
                )

        if phase_before is Phase.DAY_TARGET:
            self._resolve_day_targets(actions)
        elif phase_before is Phase.NOMINATION:
            self._resolve_nominations(actions)
        elif phase_before is Phase.RUNOFF:
            self._resolve_runoff(actions)
        elif phase_before is Phase.DEFENCE_VOTE:
            self._resolve_defence(actions)
        elif phase_before is Phase.NIGHT:
            self._resolve_night(actions)
        else:
            raise AssertionError(f"unhandled phase: {phase_before}")

        rewards = self._rewards()
        return Transition(
            phase_before=phase_before,
            phase_after=self.state.phase,
            events=tuple(self.state.events[event_start:]),
            rewards=rewards,
            terminated=self.state.terminated,
            truncated=self.state.truncated,
            winner=self.state.winner,
        )

    def _decision_actors(self) -> dict[PlayerId, DecisionKind]:
        alive = self._alive_ids()
        if self.state.phase is Phase.DAY_TARGET:
            return {pid: DecisionKind.DAY_TARGET for pid in alive}
        if self.state.phase is Phase.NOMINATION:
            return {pid: DecisionKind.NOMINATION for pid in alive}
        if self.state.phase is Phase.RUNOFF:
            excluded = set(self.state.locked_defendants) | set(
                self.state.runoff_candidates
            )
            return {pid: DecisionKind.RUNOFF for pid in alive if pid not in excluded}
        if self.state.phase is Phase.DEFENCE_VOTE:
            defendants = set(self.state.defendants)
            return {
                pid: DecisionKind.DEFENCE_VOTE
                for pid in alive
                if pid not in defendants
            }
        if self.state.phase is Phase.NIGHT:
            result: dict[PlayerId, DecisionKind] = {}
            for pid in alive:
                role = self.state.players[pid].role
                if role is Role.MAFIA:
                    result[pid] = DecisionKind.MAFIA_KILL
                elif role is Role.DOCTOR:
                    result[pid] = DecisionKind.PROTECT
                elif role is Role.DETECTIVE:
                    result[pid] = DecisionKind.INVESTIGATE
            return result
        return {}

    def _resolve_day_targets(self, actions: Mapping[PlayerId, Action]) -> None:
        self.state.day_targets = {}
        for player_id in sorted(actions):
            action = actions[player_id]
            assert isinstance(action, DayTarget)
            self.state.day_targets[player_id] = action.targets
            self._emit(
                "day_targets",
                Visibility.PUBLIC,
                actor=player_id,
                payload={"targets": action.targets},
            )
        self.state.phase = Phase.NOMINATION

    def _resolve_nominations(self, actions: Mapping[PlayerId, Action]) -> None:
        totals: Counter[PlayerId] = Counter()
        self.state.nomination_votes = {}
        for player_id in sorted(actions):
            action = actions[player_id]
            assert isinstance(action, NominationVote)
            self.state.nomination_votes[player_id] = action.targets
            totals.update(action.targets)
            self._emit(
                "nomination_vote",
                Visibility.PUBLIC,
                actor=player_id,
                payload={"targets": action.targets},
            )
        self.state.original_nomination_totals = {
            pid: totals[pid] for pid in self._alive_ids()
        }
        alive = self._alive_ids()
        ranked_candidates = sorted(alive, key=lambda pid: (-totals[pid], pid))
        cutoff = totals[ranked_candidates[1]]
        locked = tuple(sorted(pid for pid in alive if totals[pid] > cutoff))
        boundary = tuple(sorted(pid for pid in alive if totals[pid] == cutoff))
        open_seats = 2 - len(locked)
        if len(boundary) == open_seats:
            self._set_defendants(locked + boundary)
            return
        self.state.locked_defendants = locked
        self.state.runoff_candidates = boundary
        self.state.runoff_open_seats = open_seats
        self.state.phase = Phase.RUNOFF
        self._emit(
            "runoff_started",
            Visibility.PUBLIC,
            payload={
                "locked": locked,
                "candidates": boundary,
                "open_seats": open_seats,
            },
        )

    def _resolve_runoff(self, actions: Mapping[PlayerId, Action]) -> None:
        totals: Counter[PlayerId] = Counter()
        for player_id in sorted(actions):
            action = actions[player_id]
            assert isinstance(action, RunoffVote)
            totals.update(action.targets)
            self._emit(
                "runoff_vote",
                Visibility.PUBLIC,
                actor=player_id,
                payload={"targets": action.targets},
            )
        incoming = Counter(
            target
            for targets in self.state.day_targets.values()
            for target in targets
        )
        ranking = sorted(
            self.state.runoff_candidates,
            key=lambda pid: (
                -totals[pid],
                -incoming[pid],
                -self.state.original_nomination_totals.get(pid, 0),
                pid,
            ),
        )
        selected = tuple(ranking[: self.state.runoff_open_seats])
        self._emit(
            "runoff_resolved",
            Visibility.PUBLIC,
            payload={
                "totals": dict(totals),
                "ranking": tuple(ranking),
                "selected": selected,
            },
        )
        self._set_defendants(self.state.locked_defendants + selected)

    def _set_defendants(self, defendants: tuple[PlayerId, ...]) -> None:
        if len(defendants) != 2 or len(set(defendants)) != 2:
            raise RuntimeError(f"expected two distinct defendants, got {defendants}")
        self.state.defendants = tuple(defendants)
        self.state.phase = Phase.DEFENCE_VOTE
        self._emit(
            "defendants_selected",
            Visibility.PUBLIC,
            payload={"defendants": self.state.defendants},
        )

    def _resolve_defence(self, actions: Mapping[PlayerId, Action]) -> None:
        totals: Counter[PlayerId] = Counter()
        for player_id in sorted(actions):
            action = actions[player_id]
            assert isinstance(action, DefenceVote)
            totals[action.target] += 1
            self._emit(
                "defence_vote",
                Visibility.PUBLIC,
                actor=player_id,
                payload={"target": action.target},
            )
        first, second = self.state.defendants
        eliminated: PlayerId | None = None
        if totals[first] > totals[second]:
            eliminated = first
        elif totals[second] > totals[first]:
            eliminated = second
        self._emit(
            "defence_result",
            Visibility.PUBLIC,
            payload={"totals": dict(totals), "eliminated": eliminated},
        )
        if eliminated is not None:
            self._eliminate(eliminated, DeathPhase.DAY)
        if self._check_victory():
            return
        self.state.phase = Phase.NIGHT

    def _resolve_night(self, actions: Mapping[PlayerId, Action]) -> None:
        mafia_votes: dict[PlayerId, PlayerId] = {}
        protected: set[PlayerId] = set()
        investigations: list[tuple[PlayerId, PlayerId]] = []
        for player_id in sorted(actions):
            action = actions[player_id]
            if isinstance(action, MafiaKillVote):
                mafia_votes[player_id] = action.target
            elif isinstance(action, Protect):
                protected.add(action.target)
                self._emit(
                    "protection_selected",
                    Visibility.PRIVATE,
                    actor=player_id,
                    recipients=(player_id,),
                    payload={"target": action.target},
                )
            elif isinstance(action, Investigate):
                investigations.append((player_id, action.target))
            else:
                raise AssertionError(f"unexpected night action: {action!r}")

        if mafia_votes:
            self._emit(
                "mafia_kill_ballots",
                Visibility.MAFIA,
                recipients=self._mafia_ids(),
                payload={"ballots": dict(mafia_votes)},
            )
            counts = Counter(mafia_votes.values())
            highest = max(counts.values())
            tied = sorted(pid for pid, count in counts.items() if count == highest)
            attacked = self._rng.choice(tied)
            self._emit(
                "mafia_target_selected",
                Visibility.MAFIA,
                recipients=self._mafia_ids(),
                payload={"target": attacked, "tied": tuple(tied)},
            )
            if attacked not in protected:
                self._eliminate(attacked, DeathPhase.NIGHT)

        for detective, target in investigations:
            alignment = self.state.players[target].alignment
            self._emit(
                "investigation_result",
                Visibility.PRIVATE,
                actor=detective,
                recipients=(detective,),
                payload={"target": target, "alignment": alignment.value},
            )

        if self._check_victory():
            return
        if self.state.day >= self.config.max_days:
            self.state.truncated = True
            self.state.phase = Phase.TERMINAL
            self._emit("game_truncated", Visibility.PUBLIC, payload={"reason": "max_days"})
            return
        self.state.day += 1
        self.state.phase = Phase.DAY_TARGET
        self.state.day_targets = {}
        self.state.nomination_votes = {}
        self.state.original_nomination_totals = {}
        self.state.locked_defendants = ()
        self.state.runoff_candidates = ()
        self.state.runoff_open_seats = 0
        self.state.defendants = ()
        self._announce_day()

    def _announce_day(self) -> None:
        total_eliminated = len(self.state.eliminations)
        total_mafia = sum(
            self.state.players[item.player_id].alignment is Alignment.MAFIA
            for item in self.state.eliminations
        )
        mafia_delta = total_mafia - self.state.announced_mafia
        unannounced = self.state.eliminations[self.state.announced_eliminated :]
        day_deaths = [item for item in unannounced if item.phase is DeathPhase.DAY]
        for item in unannounced:
            if item.phase is DeathPhase.NIGHT:
                self.state.revealed_alignments[item.player_id] = Alignment.CITIZEN
        if len(day_deaths) == 1:
            self.state.revealed_alignments[day_deaths[0].player_id] = (
                Alignment.MAFIA if mafia_delta == 1 else Alignment.CITIZEN
            )
        elif day_deaths:
            raise RuntimeError("MVP rules allow at most one Day elimination per announcement")
        self.state.announced_eliminated = total_eliminated
        self.state.announced_mafia = total_mafia
        self._emit(
            "day_announcement",
            Visibility.PUBLIC,
            payload={
                "cumulative_eliminated": total_eliminated,
                "cumulative_mafia": total_mafia,
            },
        )

    def _eliminate(self, player_id: PlayerId, phase: DeathPhase) -> None:
        player = self.state.players[player_id]
        if not player.alive:
            raise RuntimeError(f"player {player_id} is already dead")
        player.alive = False
        self.state.eliminations.append(Elimination(player_id, self.state.day, phase))
        self._emit(
            "player_eliminated",
            Visibility.PUBLIC,
            payload={"player_id": player_id, "phase": phase.value},
        )

    def _check_victory(self) -> bool:
        mafia = sum(
            player.alive and player.alignment is Alignment.MAFIA
            for player in self.state.players.values()
        )
        citizens = sum(
            player.alive and player.alignment is Alignment.CITIZEN
            for player in self.state.players.values()
        )
        winner: Alignment | None = None
        if mafia == 0:
            winner = Alignment.CITIZEN
        elif mafia >= citizens:
            winner = Alignment.MAFIA
        if winner is None:
            return False
        self.state.winner = winner
        self.state.terminated = True
        self.state.phase = Phase.TERMINAL
        self._emit("game_ended", Visibility.PUBLIC, payload={"winner": winner.value})
        return True

    def _rewards(self) -> dict[PlayerId, float]:
        if not self.state.terminated:
            return {pid: 0.0 for pid in self.state.players}
        assert self.state.winner is not None
        return {
            pid: 1.0 if player.alignment is self.state.winner else -1.0
            for pid, player in self.state.players.items()
        }

    def _linked_players(
        self, player_id: PlayerId, candidates: tuple[PlayerId, ...]
    ) -> tuple[PlayerId, ...]:
        own_targets = set(self.state.day_targets.get(player_id, ()))
        return tuple(
            candidate
            for candidate in candidates
            if candidate != player_id
            and (
                candidate in own_targets
                or player_id in self.state.day_targets.get(candidate, ())
            )
        )

    def _alive_ids(self) -> tuple[PlayerId, ...]:
        return tuple(
            pid for pid, player in self.state.players.items() if player.alive
        )

    def _mafia_ids(self) -> tuple[PlayerId, ...]:
        return tuple(
            pid
            for pid, player in self.state.players.items()
            if player.role is Role.MAFIA
        )

    def _event_visible_to(self, event: Event, player_id: PlayerId) -> bool:
        if event.visibility is Visibility.PUBLIC:
            return True
        if event.visibility is Visibility.ENGINE:
            return False
        if event.visibility is Visibility.PRIVATE:
            return player_id in event.recipients
        return self.state.players[player_id].alignment is Alignment.MAFIA

    def _emit(
        self,
        event_type: str,
        visibility: Visibility,
        *,
        actor: PlayerId | None = None,
        recipients: tuple[PlayerId, ...] = (),
        payload: Mapping[str, object] | None = None,
    ) -> None:
        self.state.events.append(
            Event(
                sequence=len(self.state.events),
                event_type=event_type,
                day=self.state.day,
                visibility=visibility,
                actor=actor,
                recipients=recipients,
                payload=payload or {},
            )
        )
