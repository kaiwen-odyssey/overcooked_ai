# NEXUS Burger MARL Lab

BP-ready interactive visualization for the Overcooked burger multi-agent task.
It demonstrates the authoritative kitchen state, one-to-four agent scaling, a
MAPPO/CTDE training contract, failure recovery, and sim-to-real-to-sim replay.

Single-agent standard, dirty-plate, and fire-recovery scenarios replay the
accepted PPO checkpoint with deterministic masked-argmax actions. Multi-agent
scaling remains visibly labeled demonstration data until MAPPO training is
complete.

## One-command WebUI

Requires Node.js 22.13.1 or newer. From the repository root:

```bash
./burger_lab/run-webui.sh
```

Open <http://localhost:3000>. The launcher installs locked dependencies when
needed, creates a standalone production build, and starts it.

To use another port:

```bash
BURGER_WEBUI_PORT=53242 ./burger_lab/run-webui.sh
```

## Development

```bash
cd burger_lab
npm ci
npm run dev
```

The development server prints its local URL. Before submitting changes, run:

```bash
npm run lint
npm test
npm run smoke:webui
```

`npm run build` also creates `dist/standalone/`. It can be copied to another
machine with Node.js 22.13.1+ and started without the source tree:

```bash
node dist/standalone/server.js
```

The Burger WebUI GitHub Actions check uploads this standalone directory as a
downloadable PR artifact after it verifies the live page and PPO manifest.

## Main controls

- inspect the Cramped Galley PPO replay and one-to-four agent scaling study;
- switch among standard, dirty-plate, and fire-recovery PPO starts;
- play, pause, reset, and change replay speed;
- watch the policy extinguish a grill fire and resume production;
- inspect MAPPO/CTDE settings and reward design;
- export a JSON replay summary.
