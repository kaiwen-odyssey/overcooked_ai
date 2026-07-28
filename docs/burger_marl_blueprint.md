# Burger MARL task blueprint

This document defines the first BP-ready version of the Overcooked burger
extension. The interactive visualization lives in `burger_lab/`. Its current
replay is deterministic demo data; the contracts below are intended for the
real simulator and MAPPO implementation.

## 1. Authority boundary and anti-exploit contract

The Python `BurgerEnv` in
`burger_marl.env` is the only authority for state transitions,
events, rewards, and terminal status. The browser is an untrusted renderer:
it may replay environment frames, but it cannot submit reward, mutate world objects,
advance a timer, or declare a delivery.

Every joint step follows the standalone burger environment order:

1. validate all seven-action inputs and reject malformed joint actions;
2. resolve cardinally adjacent `PICK_DROP` or `PROCESS` in stable
   north/south/east/west target order and then stable player-index order;
3. resolve movement and collisions simultaneously;
4. advance grill and plate-return timers exactly once;
5. validate all state invariants;
6. derive reward only from the events produced by that transition.

The following invariants are checked after every training step:

- one to four agents, unique walkable positions, and cardinal orientation;
- no diagonal movement, same-cell endpoint, head-on swap, or station traversal;
- one object per hand and one object per counter;
- atomic pickup/drop, so simultaneous agents cannot duplicate an object;
- a clean or partially filled plate is one composite physical object: an agent
  may carry that plate and add counter or grill ingredients to it, while the
  same plate identity remains in hand;
- cooked beef cannot be held barehanded: it must move directly from grill or
  counter onto a clean or partially assembled plate;
- exact physical plate conservation across hands, visible counters, service
  return, sink, and clean rack;
- no preparation inventory or virtual assembly state: every extracted loose
  ingredient occupies one visible hand or counter position, while every
  partial burger is one plate object in a visible hand or counter position;
- grill, sink, and plate-return timer bounds;
- at most one wash tick per joint environment step; any adjacent empty-handed
  agent may continue an interrupted wash without a hidden owner reservation;
- a washed plate enters the final processing agent's empty hand, and only a
  later `PICK_DROP` at the plate rack adds it back to the visible stack;
- exactly one extinguisher across its station, agent hands, and counters;
- completed food can be rewarded only once because delivery consumes the held
  plated burger in the same transaction;
- trash disposal is adjacent interaction only: loose food is deleted, loaded
  plates become clean plates in hand, and dirty/clean plates or extinguishers
  cannot disappear;
- a SHA-256 state digest on every transition for replay-integrity checks.

Reference training uses the standalone `strict_agent_obstacles: false`
baseline: duplicate endpoints or a head-on swap cancel only the conflicting
agents, while unrelated agents and legal followers continue. A
stricter occupied-at-step-start mode remains an explicitly labeled robustness
ablation.

The contract suite in `testing/burger_mdp_test.py` includes adversarial
simultaneous pickup, duplicate counter ingredient, repeated delivery, double-washing,
collision/swap, forged plate inventory, timer-boundary, illegal-action, and
1,000-step randomized invariant tests. These tests are release gates for any
environment change.

## 2. Task loop

One episode lasts 180 simulated seconds (429 control steps at 0.42 seconds per
step) and supports one to four agents.

1. Fetch a bun, raw beef, and lettuce. Every fetched loose ingredient remains
   visible in one agent hand, one counter cell, or the grill; it is never added
   to an off-grid stock count.
2. Cook beef on a grill for 24 environment steps (10.08 seconds). Once ready,
   it has a 16-step pickup window (6.72 seconds); missing that window burns the
   patty, starts a fire, and locks the grill until a valid extinguisher
   interaction.
3. Take one clean plate from the visible rack stack, decrementing that stack,
   then carry it to the ingredient dispensers, counters, and grill. Each
   successful adjacent interaction transfers exactly one ingredient onto that
   same plate; bun and lettuce may also be carried loose, but cooked beef may
   leave the grill only by being collected onto a plate.
4. Deliver the completed burger to the delivery hatch. The complete meal and
   plate leave the kitchen immediately.
5. The same physical plate returns dirty after 12 steps (5.04 seconds) at a
   separate dish-return hatch and joins its integer queue.
6. Move exactly one dirty plate to the sink and spend 10 separate interaction
   steps washing it. The clean plate finishes in that agent's hand and may be
   returned to the rack, where up to all four clean plates can stack. Food and
   loaded plates remain single objects on single-slot counters.
7. If the grill catches fire, fetch the sole physical extinguisher and interact from the
   any cardinally adjacent floor cell.
8. If an ingredient or plated meal must be abandoned, carry it to the trash
   station. A loaded plate is emptied but retained; disposal earns no positive
   event reward and normally loses potential plus the per-step time cost.

The primary objective is the number of correct deliveries in 180 seconds.
There is no individual score.

## 3. Three evaluation layouts

| Layout | Based on | Coordination pressure | Expected scaling behavior |
| --- | --- | --- | --- |
| Cramped Galley | Cramped Room | One grill and a narrow central lane | Strong 1→3 gain; four agents may saturate or regress |
| Split Service | Forced Coordination / Asymmetric Advantages | Prep and service zones separated by one pass-through | Handoffs are mandatory; four agents can remain useful |
| Coordination Ring | Coordination Ring | One-way loop around a central island | Role specialization helps; counter-flow causes blocking |

Every headline result must be reported per layout as well as macro-averaged.
A single aggregate can otherwise hide a policy that only works in open maps.

## 4. CTDE contract

### Decentralized actor input

Each actor receives only:

- a 9 × 9 world-aligned semantic crop (four-cell radius);
- terrain from which legal movement can be inferred;
- the actor and visible teammates, including visible held objects;
- visible station states (including cook/fire timers);
- remaining episode time;
- a learned agent-ID embedding.

No teammate intent, global position, hidden object, or global order queue is
available at execution time. Occlusion is applied before encoding.

### Action space

The seven official-gameplay-derived discrete actions are:

`NORTH`, `SOUTH`, `EAST`, `WEST`, `STAY`, `PICK_DROP`, and `PROCESS`.

`PICK_DROP` moves objects, loads stations, combines plates, and delivers.
`PROCESS` advances washing and operates the extinguisher. Both are
context-sensitive adjacent actions. Facing is deliberately omitted from this
first learnable abstraction. Throw and dash remain a later physics
extension rather than being approximated inside this action space.

### Centralized critic input

The critic uses the full semantic grid, all agent positions and orientations,
inventories, station timers, fire state, pending plate-return timers, and
active-agent mask during training only. These fields distinguish
states that can have different next-state distributions, so the critic input
is Markov for the implemented dynamics.

Actors share weights. Inactive slots are excluded from policy loss and action
sampling. The team critic is trained once per environment transition rather
than once per active actor, preventing agent-count-dependent value weighting.

### Implemented trainer interface

`burger_marl.mappo_env.BurgerMAPPOEnv` provides four fixed
agent slots for vectorized PPO/MAPPO runners. Active actors receive a
26-channel 9 × 9 world-aligned crop; inactive slots receive zeros and may submit
only `STAY`. The centralized state is returned through a separate critic-only
array. The environment repeats one validated team reward across active agents,
returns explicit active masks, and never places reward or hidden global state
inside actor observations.

`burger_marl.training` is the executable PyTorch trainer. It
runs ordinary PPO only when `num_agents=1`; MAPPO mode requires at least two
agents and cannot be selected with a misleading one-agent label. Checkpoints
store actor and critic separately, so the stage-one actor can warm-start the
stage-two shared actor while the centralized critic is initialized fresh.

Before actor or critic optimization begins, the trainer runs an executable
preflight against the authoritative environment. It verifies the index and
physical effect of all seven actions, exercises washing and fire suppression,
checks every reward component, checks one- through four-agent masks, and
confirms that vectorized masks change immediately after state transitions.
The trainer fails closed on a non-binary or empty active mask. A passed report
is written to `preflight.json` beside the checkpoints.

## 5. First one-map training milestone

The first controlled experiment intentionally uses only **Cramped Galley**:

1. Train one-agent PPO for 3 million environment transitions from random
   initialization.
2. Accept stage one only if deterministic fixed-start evaluation averages at
   least one valid delivery per 180-second episode and exceeds a random policy.
3. Warm-start the shared two-agent actor from the accepted one-agent actor.
   Initialize a fresh centralized critic and train two-agent MAPPO for
   9 million environment transitions.
4. Accept stage two only if it preserves valid deliveries and exceeds the
   one-agent deliveries-per-minute result. Report collisions alongside
   throughput so apparent speedup cannot hide congestion.

The 3M + 9M split preserves the proposed 12-million-step budget while making
the causal question explicit: first learn the recipe, then learn cooperation.
If the stage-one gate fails, do not start the expensive MAPPO run; diagnose
exploration, potential shaping, and task observability first.

Reference commands:

```bash
.venv/bin/python -m pip install ".[burger_train]"

.venv/bin/python -m burger_marl.training \
  --algorithm ppo --num-agents 1 --total-env-steps 3000000 \
  --num-envs 64 --rollout-length 256 --output-dir runs/burger

.venv/bin/python -m burger_marl.training \
  --algorithm mappo --num-agents 2 --total-env-steps 9000000 \
  --num-envs 64 --rollout-length 256 --output-dir runs/burger \
  --actor-init runs/burger/ppo_1agent_seed20260728/final.pt
```

The fixed hyperparameter records are
`configs/burger_ppo_1agent.yaml` and
`configs/burger_mappo_2agent.yaml`. Smoke tests should use a separate output
directory and must not be presented as learned-policy results.

Every rollout records the frequency of all seven actions, the ratio of joint
all-`STAY` steps, the ratio of joint steps that produce neither motion nor an
agent-attributed environment event, the mean number of legal actions, and the
total of every reward component. If the all-`STAY` ratio reaches `0.98` or the
all-no-op ratio reaches `0.995` for three consecutive updates, training aborts
before applying the next PPO update. Deterministic evaluation reports the same
freeze indicators; stage acceptance additionally requires all-`STAY < 0.95`
and all-no-op `< 0.98`.

## 6. PPO/MAPPO configuration

- Shared two-layer CNN + 256-unit MLP actor with agent-ID embedding.
- Separate centralized 256-unit MLP critic.
- Adam learning rate `3e-4`.
- `gamma=0.99`, `gae_lambda=0.95`, PPO clip `0.20`.
- Value clip `0.20`, entropy coefficient `0.01`.
- Rollout length 256, 64 parallel environments, 4 PPO epochs, 8 mini-batches.
- Gradient norm clipping at 10.
- Curriculum: one agent until recipe competence, two agents until handoffs are
  stable, then sample two/three/four agents uniformly.
- Layout curriculum: Cramped Galley → Split Service → mixed three-layout
  training with 20% procedural perturbations.
- Evaluation: deterministic actor, 100 episodes per layout/agent-count/seed,
  five seeds, 95% bootstrap confidence intervals.

The feed-forward actor is the auditable first baseline. A GRU/attention model
is a later ablation, not silently mixed into the first one-map comparison.

## 7. Reward

Use one shared team reward:

```text
r_t = 20 * correct_delivery
    -  5 * fire_started
    - 0.05 * blocked_or_collision
    - 0.01
    + gamma * Phi(s_{t+1}) - Phi(s_t)
```

`Phi` is a bounded potential over recipe progress: raw beef contributes 0.5,
cooked beef contributes 1.0, and bun, lettuce, a visible plate, and complete
assembly add their documented state progress. The
potential term must be transition-based, not a repeated state bonus, so agents
cannot farm intermediate steps.

Invalid or repeated interactions receive zero event reward. Do not add a
handoff bonus to the final training objective: the policy must value handoffs
only through their effect on valid recipe progress and delivery throughput.
The shaping `gamma` is materialized from the same `TrainConfig.gamma` used by
PPO, and the terminal potential is forced to zero. This preserves
potential-based policy invariance: pickup/drop cycles telescope instead of
creating a repeatable positive-reward loop.

## 8. Required metrics

### Primary

- correct deliveries per 180 seconds;
- deliveries per minute;
- correct-order completion rate;
- mean and p90 order latency.

### Coordination and efficiency

- collision or blocked moves per 100 steps;
- agent idle-time ratio;
- station utilization and queue time;
- handoff success rate and mean handoff wait;
- workload Gini coefficient;
- role-specialization score: normalized mutual information between agent ID
  and station/action category;
- marginal contribution of the N-th agent:
  `(throughput_N - throughput_{N-1}) / throughput_{N-1}`.

### Robustness

- fire incidence per episode;
- fire recovery success rate and p90 recovery time;
- delivery degradation under a fire;
- deadlock rate and longest no-progress interval;
- generalization to perturbed cook/wash/return times;
- cross-play score when agents come from different seeds.

The preferred BP chart plots throughput and collision rate together for one to
four agents. The result should visibly communicate useful scaling followed by
congestion-driven saturation, without assuming in advance that four agents
must be worse.

## 9. Failure-case learning loop

Record a compact trajectory window from 10 seconds before to 15 seconds after:

- fire or overcook;
- deadlock lasting more than 5 seconds;
- wrong delivery or dropped handoff;
- order latency above the training p95;
- critic high-error or actor low-confidence decisions.

Cluster failures by layout region, station state, active roles, and event type.
For every selected cluster, generate 64 simulator variants by perturbing agent
spawn, cook time, plate return, friction, sensor occlusion, and order timing.
Reserve 30% of training batches for prioritized failure variants and keep a
fixed, unseen failure suite for regression testing.

## 10. Replay schema

Every visualization frame should be derived from environment state, not from
free-running UI animation:

```json
{
  "t": 46.0,
  "map": "split_service",
  "agents": [
    {
      "id": 0,
      "position": [4, 3],
      "orientation": "EAST",
      "held_object": "plate_cooked_beef",
      "action": "PICK_DROP",
      "role_probe": "cook"
    }
  ],
  "stations": {
    "grill_0": {"status": "cooking", "progress": 0.73},
    "sink_0": {"status": "washing", "progress": 0.41},
    "trash_0": {"status": "available"}
  },
  "orders": [{"recipe": "classic_burger", "age": 21.4}],
  "team_reward": 1.0,
  "events": [{"type": "handoff", "from": 1, "to": 2}]
}
```

The web demo should consume this schema unchanged when connected to the Python
runner. Until that adapter is connected, the browser animation must remain
visibly labeled as a deterministic demo replay and must never be reported as a
MAPPO result.
