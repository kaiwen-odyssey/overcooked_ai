# NEXUS Burger MARL Lab

BP-ready interactive visualization for the Overcooked burger multi-agent task.
It demonstrates three coordination layouts, one-to-four agent scaling, a
MAPPO/CTDE training contract, failure injection, and sim-to-real-to-sim replay.

The current browser animation uses deterministic scripted trajectories and is
visibly labeled as demo data. It does not claim to be a trained checkpoint.

## Run

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Open the local URL printed by the development server. A production-compatible
build is created with:

```bash
npm run build
```

The connected Python simulator should emit the frame contract documented in
`../docs/burger_marl_blueprint.md`. The interface can then replace its scripted
frame generator without changing the visual components.

## Main controls

- switch among Cramped Galley, Split Service, and Coordination Ring;
- compare one to four active agents;
- play, pause, reset, and change replay speed;
- inject an overcooked-grill failure and watch recovery;
- inspect MAPPO/CTDE settings and reward design;
- export a JSON replay summary.
