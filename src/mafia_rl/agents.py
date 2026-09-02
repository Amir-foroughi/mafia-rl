"""Baseline agents that depend only on decision requests and observations."""

from __future__ import annotations

import random
from typing import Protocol

from .models import Action, DecisionRequest, Observation


class Agent(Protocol):
    def act(self, observation: Observation, request: DecisionRequest) -> Action:
        """Return one of ``request.legal_actions``."""


class RandomAgent:
    """Uniformly sample a complete legal domain action."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: Observation, request: DecisionRequest) -> Action:
        del observation
        if not request.legal_actions:
            raise RuntimeError("received a decision request with no legal actions")
        return self._rng.choice(request.legal_actions)
