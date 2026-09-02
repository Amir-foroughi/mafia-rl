# Modular Mafia Reinforcement-Learning Project

## Approved MVP

Build a configurable, headless Python Mafia engine separated from agents and
reinforcement-learning libraries. The initial roster is three Mafia, one Doctor,
one Detective, and five normal Citizens.

The game begins with a reveal-only Night 0, followed by Day 1 / Night 1 through
Day 10 / Night 10. Citizens win when no Mafia remain; Mafia win at parity; an
unresolved game after Night 10 is a draw.

During each Day, every living player targets between one and the publicly known
number of living Mafia, capped by the available other players. Targets are
distinct, public, and may include Mafia teammates. Every nomination ballot has
two distinct choices. Target-linked players must be used first; missing choices
are filled from any living player, including the voter. A cutoff tie triggers one
runoff. A remaining tie is resolved by runoff votes, incoming Day targets,
original nominations, and ascending seat ID. The two defendants abstain from the
final vote; every other player chooses one defendant, and an equal result
eliminates nobody.

At Night, each living Mafia secretly votes for a living non-Mafia target. The
plurality target is attacked, with seeded randomness for a tied plurality. The
Doctor may protect any living player, including themself and repeated targets.
The Detective may repeatedly investigate another living player and privately
learns Mafia or Not Mafia. A matching protection prevents death without save
confirmation.

At the beginning of every Day, the director announces cumulative deaths and
cumulative Mafia deaths since Day 1. Player identities and death timing are
public, exact roles remain hidden, and publicly inferable eliminated alignments
are included in observations. Since night victims are necessarily Citizens, the
alignment of a Day-eliminated player is inferable from the cumulative Mafia
delta.

## Architecture

- A standard-library core owns configuration, state, typed actions, events,
  observation projection, phase resolution, victory, and rewards.
- Agents receive only observations and legal actions; they never receive the
  authoritative state.
- All randomness is seeded and all resolutions are replayable from the event
  log.
- Invalid actions raise errors rather than being silently repaired.
- Gymnasium, PettingZoo, PyTorch, and learning algorithms remain outside the
  engine and are introduced through adapters in later phases.

## Learning progression

1. Deterministic engine and tests.
2. Seeded random agents, headless simulation, and statistics.
3. Rule-based agents with agent-owned trust beliefs.
4. RL observations, discrete action catalogs, and action masks.
5. A focal normal-Citizen MaskablePPO baseline.
6. Role policies, opponent leagues, and multi-agent self-play.
7. Recurrent policies and centralized-training/decentralized-execution trials.

Trust is evidence rather than a game mechanic. The first rule baseline will use
weak target evidence, medium nomination evidence, and strong final-vote evidence.
Supporting an eliminated Mafia can increase trust, supporting an eliminated
Citizen can reduce it, and Mafia may exploit this by voting against teammates.
Detective results remain private belief overrides.

## Quality gates

Tests cover configuration, phase transitions, mandatory voting, nomination
fallbacks, runoff ranking, defence ties, night resolution, cumulative
announcements, visibility, terminal conditions, determinism, replay, and
submission-order invariance. Each phase is reviewed and tested before its commit.
