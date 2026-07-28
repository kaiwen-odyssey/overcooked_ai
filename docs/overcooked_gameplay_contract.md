# Official-gameplay-derived standalone burger contract

This contract no longer treats the repository's original soup environment as
an implementation dependency. `BurgerGridworld` and `burger_marl.actions` are
the standalone executable source of truth. The commercial game informs the
interaction design; the burger, fire, delayed-return, and trash systems are
explicit research-task rules.

## Reference mechanics

The first milestone implements the stable cooperative kitchen core:

1. Each player selects exactly one of seven actions per environment step:
   north, south, east, west, stay, pick/drop, or process.
2. There is no diagonal movement. Orientation is retained for rendering and a
   later facing-fidelity extension, but is not required for milestone-one work.
3. `PICK_DROP` and `PROCESS` select one valid cardinally adjacent station in
   stable north/south/east/west order. Pick/drop moves physical objects or
   completes contextual delivery; process advances washing or fire suppression.
4. Interactions are resolved before movement and in stable player-index order.
   Later players see mutations made by earlier players in the same joint step.
5. Players can stand only on floor. Counters and stations are not walkable.
6. A counter stores at most one object. A player hand stores at most one
   physical object; a plate and its contents are one composite object.
7. Objects can move only through a successful interaction between a hand and a
   counter or station. They cannot be dropped on floor.
8. Movement is simultaneous. If proposed endpoints overlap, or two players
   swap positions head-on, only the agents participating in the conflict are
   cancelled; unrelated agents continue on that joint step.
9. Following into a cell vacated during the same joint step is enabled for the
   baseline. `strict_agent_obstacles` provides the occupied-at-step-start
   ablation.
10. Autonomous environment effects advance exactly once after interactions and
    movement.
11. Sparse task reward comes from a valid delivery transition, not from a
    policy claim, animation frame, or UI counter.

The original benchmark task is soup delivery: collect ingredients, place them
in a pot, wait for cooking, collect ready soup with a dish, and serve it. The
burger recipe is not part of the upstream benchmark.

## Explicit burger extension

The burger task preserves all reference movement and interaction mechanics,
while replacing the recipe state machine:

1. bun, lettuce, raw beef, and clean plates are physical objects;
2. beef cooks autonomously for 24 control steps, remains ready for a 16-step
   pickup window, and then burns and locks the grill until extinguished;
3. a held plate may collect bun or lettuce directly from a dispenser, visible
   ingredients from a counter, and cooked beef directly from the grill, one
   adjacent interaction at a time;
4. cooked beef can never be held barehanded. It remains on the grill or counter
   until an agent carrying a clean or partially assembled plate collects it;
5. there is no preparation inventory or virtual assembly station. Every
   extracted loose ingredient occupies one visible hand, counter, or grill
   position; every partial burger is a single plate object held by an agent or
   occupying exactly one counter cell;
6. only a plate containing bun, lettuce, and cooked beef is deliverable;
7. the delivery hatch accepts only complete plated burgers; delivery consumes
   the meal and transfers the same physical plate into a 12-step delayed return
   queue;
8. the separate dish-return hatch is the only station that releases returned
   dirty plates. Each returned plate must be carried to the sink and washed for the complete
   configured duration before becoming a clean plate in the washing agent's
   hand. Taking a clean plate decrements the visible rack stack; returning one
   increments it. Up to all four clean plates may share the rack, while every
   counter remains single-slot and food can never stack;
9. exactly one physical extinguisher exists. Its station is empty while the
   extinguisher is held or rests on a counter. Only an adjacent agent holding
   that extinguisher can clear a burning grill with `PROCESS`; fire recovery
   is an environment event and cannot be triggered by UI state.
10. an adjacent interaction with the trash station may delete loose food or
   empty the food from a loaded plate; the physical plate remains in hand as
   a clean plate. Clean plates, dirty plates, and extinguishers cannot be
   deleted at the trash station.

Throwing and dashing from Overcooked 2 are intentionally excluded from the
first controlled milestone. They require a separately validated trajectory
and collision model and must not be silently represented as ordinary movement.

## PPO and MAPPO boundary

`BurgerMAPPOEnv` exposes four fixed agent slots:

- actor observation: a 26-channel, 9 by 9 world-aligned semantic crop, including
  distinct trash, delivery, and dish-return terrain channels;
- critic observation: a separate Markov global state, including orientations,
  pending plate-return timers, sink owner, and active-agent mask;
- action: one integer in the standalone seven-action set;
- action mask: `STAY`, walkable cardinal moves, and only those `PICK_DROP` or
  `PROCESS` actions for which a locally visible adjacent station can produce a
  valid physical event;
- reward: the same validated team reward for every active agent;
- inactive slots: zero observations and only `STAY` available;
- renderer: no state or reward mutation capability.

Every returned actor observation is constructed from the local crop after
occlusion. It contains no reward, teammate intent, hidden global inventory, or
centralized critic state. The adapter returns deep copies and owns the
authoritative environment state.

## Reward-hacking release gates

An environment revision is blocked if any of these tests fail:

- illegal or non-integer action rejection;
- cardinal-adjacent interaction independent of facing;
- cooked-beef plate requirement at grill and counter;
- simultaneous pickup atomicity;
- one-object-per-counter and one-object-per-hand;
- no floor drop or silent deletion;
- trash disposal has no positive event reward and never deletes a plate;
- plate conservation;
- duplicate-component rejection;
- repeated-delivery rejection;
- full-duration cooking and washing;
- no multi-agent double washing;
- exhaustive collision checks, including unrelated-agent progress;
- deterministic state digest under identical trajectories;
- local observation non-interference from hidden global changes;
- equal shared reward for active agents;
- no positive reward from repeated invalid interaction;
- action-mask decisions use only local-visible state and sampled PPO actions
  always belong to the recorded mask;
- vectorized action masks are replaced after every environment step and reset,
  never cached from an earlier state;
- all seven action indices pass executable effect checks before optimization;
- every sampled team reward equals the sum of `correct_delivery`,
  `fire_started`, `collision`, `time_step`, and potential shaping;
- potential shaping uses the exact PPO discount factor, and sustained
  all-`STAY` or all-no-op rollouts abort training;
- randomized long-horizon invariant checks.
