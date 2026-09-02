# Simplified Mafia RL Plan

The engine treats Day targeting and voting as independent public actions. Every
living player publicly chooses zero to three distinct living targets, including
themself if desired. These choices have no direct mechanical effect except as a
deterministic runoff tiebreak.

Every living player then nominates exactly two distinct other living players,
regardless of targeting. The two highest totals enter defence. A cutoff tie
causes one runoff: locked candidates and tied candidates cannot vote, while each
remaining voter chooses enough tied candidates to fill the open seats. Remaining
ties are ranked by runoff total, incoming Day targets (including self-targets),
original nomination total, and ascending seat ID. Defendants abstain from the
mandatory final vote; a tied final vote eliminates nobody.

Night actions, private Mafia and Detective information, cumulative announcements,
victory, seeded randomness, and Day-10 truncation remain engine-owned rules. The
engine never computes trust or suspicion and never biases Detective behavior.

## Learning progression

The first experiment uses `FocalCitizenEnv`: one randomly seated normal Citizen
acts through a fixed discrete action catalog and legal-action mask while frozen,
seeded `RandomAgent` opponents control every other seat. The wrapper advances the
game until the focal player acts again and continues internally after their death.
Only the faction result is rewarded: +1 for a Citizen win, -1 for a Mafia win,
and 0 during play or on truncation.

Install the optional RL dependencies with `pip install -e .[rl]`. The adapter is
compatible with MaskablePPO's `action_masks()` convention.

After measuring the focal-Citizen baseline, add distinct Mafia, Citizen, Doctor,
and Detective policies, evaluate each on held-out seeds, alternate role training,
add historical snapshots to the opponent pool, and begin full self-play only
after every role passes its independently measured random baseline.
