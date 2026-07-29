import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import test from "node:test";
import ts from "typescript";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: {
        accept: "text/html",
        host: "localhost",
        "x-forwarded-proto": "http",
      },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

async function loadPureSimulation() {
  const ppoArtifact = JSON.parse(
    await readFile(
      new URL(
        "../public/ppo-single-agent-artifact.json",
        import.meta.url,
      ),
      "utf8",
    ),
  );
  let pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );
  pageSource = pageSource
    .replace(/^"use client";\s*/m, "")
    .replace(/^import[^;]+;\s*/gm, "");
  pageSource = pageSource.slice(
    0,
    pageSource.indexOf("export default function Home()"),
  );
  pageSource += `
globalThis.__simulation = {
  maps,
  orthogonalizePath,
  createMotionState,
  advanceMotionState,
  pointKey,
  countPhysicalPlates,
  countPhysicalExtinguishers,
};`;
  const javascript = ts.transpileModule(pageSource, {
    compilerOptions: {
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  const context = {
    console,
    ppoArtifact,
    exports: {},
    module: { exports: {} },
    require(identifier) {
      if (identifier === "react/jsx-runtime") {
        return {
          jsx() {},
          jsxs() {},
          Fragment: Symbol("Fragment"),
        };
      }
      throw new Error(`Unexpected simulation import: ${identifier}`);
    },
  };
  context.globalThis = context;
  vm.runInNewContext(javascript, context);
  return context.__simulation;
}

test("server-renders the NEXUS simulator shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html lang="zh-CN">/i);
  assert.match(html, /<title>NEXUS · 多主体汉堡协作训练平台<\/title>/i);
  assert.match(html, /NEXUS/);
  assert.match(html, /汉堡协作厨房/);
  assert.match(html, /PPO 策略回放|确定性演示回放/);
  assert.match(html, /STANDALONE BURGER CORE · 7-ACTION PPO\/MAPPO INTERFACE/);
  assert.match(html, /Cramped Galley/);
  assert.match(html, /协作规模与边际效应/);
  assert.match(html, /训练步数与每局平均送餐数/);
  assert.match(html, /5\.21(?:<!-- -->)? 份\/局/);
  assert.match(
    html,
    /标准(?:<!-- -->|\s)*5\.675(?:<!-- -->|\s)*· 脏盘(?:<!-- -->|\s)*5\.225(?:<!-- -->|\s)*· 缺盘(?:<!-- -->|\s)*4\.700(?:<!-- -->|\s)*· 过火(?:<!-- -->|\s)*5\.250/,
  );
  assert.match(html, /累计采样环境步数/);
  assert.match(html, /切换为随机位置评估/);
  assert.match(html, /单位时间送餐/);
  assert.match(html, /团队总奖励/);
  assert.match(html, /协同增益/);
  assert.match(html, /边际吞吐/);
  assert.match(html, /外围连续工作台包围中央实体岛台/);
  assert.match(html, /PPO 开局场景/);
  assert.match(html, /标准生产/);
  assert.match(html, /灭火恢复/);
  assert.match(html, /灶台起火 · 扑灭后恢复出餐/);
  assert.match(html, /脏盘回收/);
  assert.match(html, /手持脏盘 · 洗净后恢复生产/);
  assert.match(html, /POLICY ONLY/);
  assert.match(html, /桌台 \/ 灶台碰撞/);
  assert.match(html, /硬约束/);
  assert.match(html, /同格终点 \/ 迎面换位/);
  assert.match(html, /仅冲突主体等待/);
  assert.match(html, /送餐口收走成品 → 5\.0 秒后脏盘从独立出餐口退出/);
  assert.match(html, /12 步 \/ 5\.0 秒/);
  assert.match(html, /散装食材可丢弃 · 盘中内容倒空后保留实体净盘/);
  assert.match(html, /垃圾桶/);
  assert.match(html, /废弃处理/);
  assert.match(html, /1 盘 \/ 4\.2 秒/);
  assert.doesNotMatch(html, /洗碗池内有 0 个脏盘/);
  assert.match(html, /奖励结算/);
  assert.match(html, /仅环境事件/);
  assert.match(html, /每步校验 \+ 摘要/);
  assert.match(html, /实体盘子守恒/);
  assert.match(html, /4<!-- --> \/ <!-- -->4/);
  assert.match(html, /实体灭火器守恒/);
  assert.match(html, /1<!-- --> \/ 1/);
  assert.match(html, /盘架 0–4 · 食品单格/);
  assert.match(html, /网页不能修改状态或分数/);
  assert.match(html, /累计事件奖励/);
  assert.match(html, /出餐 <!-- -->0\.0<!-- --> · 起火 <!-- -->0/);
  assert.doesNotMatch(html, /交接 \+1\.5/);
  assert.match(
    html,
    /煎制 24 步（10\.1 秒）· 熟后 16 步（6\.7 秒）内必须取走/,
  );
  assert.match(html, /相邻即可 · 无需朝向/);
  assert.match(html, /↑ ↓ ← → · STAY · PICK\/DROP · PROCESS/);
  assert.match(html, /相邻即可，无需转身/);
  assert.match(html, /熟牛肉取用/);
  assert.match(html, /必须用餐盘承接/);
  assert.match(
    html,
    /散装食材占用一个桌格 · 食材源可直接加入手持餐盘/,
  );
  assert.match(
    html,
    /手持半成品可在食材源或煎台直接合并 · 全程显示餐盘/,
  );
  assert.doesNotMatch(html, /组装台/);
  assert.match(html, /SIM-TO-REAL-TO-SIM/);
  assert.doesNotMatch(html, /codex-preview/);
  assert.doesNotMatch(html, /Your site is taking shape/);
});

test("publishes a full 180-second standard PPO episode", async () => {
  const replay = JSON.parse(
    await readFile(
      new URL("../public/ppo-policy-replay.json", import.meta.url),
      "utf8",
    ),
  );

  assert.equal(replay.schema, "nexus.burger.ppo-policy-replay.v1");
  assert.equal(replay.algorithm, "PPO");
  assert.equal(replay.scenarioId, "standard");
  assert.equal(replay.curriculumStage, "random_standard_180s");
  assert.equal(replay.frames.at(-1).state.step, 429);
  assert.equal(replay.frames.length, 430);
  assert.ok(replay.deliveries >= 1);
  assert.ok(replay.washedPlates >= 1);
  assert.equal(replay.fires, 0);
  assert.ok(replay.maxStagnationSteps <= 60);
  assert.ok(
    replay.frames.some((frame) => frame.events.includes("plate_washed")),
  );
});

test("publishes successful fire and dirty-plate PPO scenarios", async () => {
  const [fire, dirty] = await Promise.all(
    ["fire", "dirty"].map(async (scenario) =>
      JSON.parse(
        await readFile(
          new URL(
            `../public/ppo-policy-replay-${scenario}.json`,
            import.meta.url,
          ),
          "utf8",
        ),
      ),
    ),
  );

  assert.equal(fire.scenarioId, "fire");
  assert.equal(fire.startStage, "fire_recovery_ready");
  assert.equal(fire.frames[0].state.grillFood, "burnt-beef");
  assert.equal(fire.frames[0].state.servedOrders, 0);
  assert.ok(fire.deliveries >= 1);
  assert.ok(fire.firesExtinguished >= 1);
  assert.ok(
    fire.frames.some((frame) =>
      frame.events.includes("fire_extinguished"),
    ),
  );
  assert.ok(
    fire.frames.some((frame) =>
      frame.events.includes("extinguisher_return"),
    ),
  );

  assert.equal(dirty.scenarioId, "dirty");
  assert.equal(dirty.startStage, "dirty_plate_carry_ready");
  assert.equal(dirty.frames[0].state.agents[0].carrying, "dirty-plate");
  assert.equal(dirty.frames[0].state.servedOrders, 0);
  assert.ok(dirty.deliveries >= 1);
  assert.ok(dirty.washedPlates >= 1);
  assert.ok(
    dirty.frames.some((frame) => frame.events.includes("plate_washed")),
  );

  for (const replay of [fire, dirty]) {
    assert.equal(replay.frames.at(-1).state.step, 429);
    assert.equal(replay.frames.length, 430);
    assert.ok(replay.maxStagnationSteps <= 60);
  }
});

test("binds every PPO replay and metric to one checkpoint artifact", async () => {
  const replayNames = ["standard", "dirty", "fire"];
  const replayEntries = await Promise.all(
    replayNames.map(async (name) => {
      const path = new URL(
        `../public/ppo-policy-replay-${name}.json`,
        import.meta.url,
      );
      const bytes = await readFile(path);
      return {
        name,
        payload: JSON.parse(bytes.toString("utf8")),
        sha256: createHash("sha256").update(bytes).digest("hex"),
      };
    }),
  );
  const artifact = JSON.parse(
    await readFile(
      new URL(
        "../public/ppo-single-agent-artifact.json",
        import.meta.url,
      ),
      "utf8",
    ),
  );

  assert.equal(artifact.acceptance.accepted, true);
  assert.equal(artifact.evaluation.episodes, 160);
  assert.equal(artifact.evaluation.positions_per_scenario, 40);
  assert.equal(artifact.evaluation.aggregate_mean_deliveries, 5.2125);
  assert.match(artifact.checkpoint.sha256, /^[0-9a-f]{64}$/);
  assert.match(artifact.checkpoint.actor_state_sha256, /^[0-9a-f]{64}$/);

  for (const { name, payload, sha256 } of replayEntries) {
    assert.equal(payload.checkpointSha256, artifact.checkpoint.sha256);
    assert.equal(
      payload.actorStateSha256,
      artifact.checkpoint.actor_state_sha256,
    );
    const evidence = artifact.webui_replays.find(
      (entry) => entry.scenario === payload.scenarioId,
    );
    assert.ok(evidence, `missing ${name} replay evidence`);
    assert.equal(evidence.sha256, sha256);
  }
});

test("publishes site-specific metadata", async () => {
  const response = await render();
  const html = await response.text();

  assert.match(html, /property="og:title"/i);
  assert.match(html, /NEXUS · Multi-Agent Emergence Lab/);
  assert.match(html, /http:\/\/localhost\/og\.png/);
  assert.match(html, /name="twitter:card" content="summary_large_image"/i);
});

test("locks counter objects behind validated context actions", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );
  const cssSource = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );

  assert.match(pageSource, /counterObjects:/);
  assert.match(pageSource, /counterMutation:/);
  assert.match(
    pageSource,
    /!\["PICK\/DROP", "PROCESS"\]\.includes\(interaction\.agent\.lastAction\)/,
  );
  assert.match(
    pageSource,
    /World state mutation requires a valid context action/,
  );
  assert.match(
    pageSource,
    /Counter object changed without a matching pickup\/drop/,
  );
  assert.match(
    pageSource,
    /Counter capacity exceeded: one object per surface cell/,
  );
  assert.match(
    cssSource,
    /\.counter-surface-item \*[\s\S]*animation: none !important;/,
  );
});

test("keeps every prepared item on a visible counter or plate", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.doesNotMatch(pageSource, /\| "assembly"/);
  assert.doesNotMatch(pageSource, /\{ kind: "assembly"/);
  assert.match(pageSource, /\[5, 7\]/);
  assert.match(pageSource, /counterObjects:/);
  assert.match(pageSource, /counterMutation:/);
});

test("renders one persistent plate beneath every plated food state", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );
  const cssSource = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );
  const platedStates = [
    "plate-bun",
    "plate-lettuce",
    "plate-cooked-beef",
    "plate-bun-lettuce",
    "plate-bun-cooked-beef",
    "plate-lettuce-cooked-beef",
    "plated-burger",
  ];

  for (const state of platedStates) {
    assert.match(
      pageSource,
      new RegExp(`"${state}": \\[`),
      `${state} must declare visible plated layers`,
    );
  }
  assert.match(pageSource, /className="food-plate-base"/);
  assert.match(pageSource, /data-plate-visible=\{platedLayers/);
  assert.match(
    pageSource,
    /className="counter-surface-item"[\s\S]*<FoodArt kind=\{counterItems\[index\]\.kind\}/,
  );
  assert.match(
    pageSource,
    /className=\{`carried-item[\s\S]*<FoodArt kind=\{carry\}/,
  );
  assert.match(cssSource, /\.food-with-visible-plate \{/);
  assert.match(cssSource, /\.food-plate-base \{/);
});

test("keeps grill timing, lockout, and fire penalty explicit", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(pageSource, /GRILL_COOK_STEPS = 24/);
  assert.match(pageSource, /GRILL_BURN_STEPS = 16/);
  assert.match(pageSource, /FIRE_STARTED_PENALTY = -5/);
  assert.match(pageSource, /fireStartedDelta = 1/);
  assert.match(pageSource, /fireExtinguishedDelta = 1/);
  assert.match(pageSource, /data-grill-usable=/);
  assert.match(pageSource, /extinguisherAtStation: true/);
  assert.match(pageSource, /countPhysicalExtinguishers\(nextState\) !== 1/);
  assert.match(pageSource, /PICK\/DROP · 应急放回净盘/);
  assert.match(pageSource, /灶台锁定 · 等待灭火器/);
});

test("adds interaction-gated trash without deleting physical plates", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );
  const cssSource = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );

  assert.match(pageSource, /\{ kind: "trash", x: 7, y: 0 \}/);
  assert.match(pageSource, /trashDiscardableItems/);
  assert.match(cssSource, /\.pictogram-trash i:nth-child\(1\) \{/);
  assert.match(cssSource, /\.pictogram-trash i:nth-child\(2\) \{/);
  assert.match(cssSource, /\.pictogram-trash i:nth-child\(3\) \{/);
  assert.match(pageSource, /loadedPlateItems\.has\(discarded\)/);
  assert.match(pageSource, /保留净盘/);
  assert.match(pageSource, /discardedDelta: 1/);
  assert.match(pageSource, /Physical plate conservation invariant violated/);
});

test("separates delivery and delayed dish return below an unobstructed board", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(pageSource, /\{ kind: "return", x: 6, y: 7 \}/);
  assert.match(pageSource, /PLATE_RETURN_DELAY_STEPS = 12/);
  assert.match(pageSource, /schedulePlateReturn: true/);
  assert.match(pageSource, /actAtStation\("return"/);
  assert.match(pageSource, /className="below-board-hud"/);
  assert.doesNotMatch(
    pageSource,
    /className="kitchen-legend"[\s\S]*备料[\s\S]*煎制[\s\S]*组装/,
  );
});

test("shows explicit dirty-dish occupancy inside the sink", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );
  const cssSource = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );

  assert.match(pageSource, /dirtyPlatesInSink: number/);
  assert.match(pageSource, /sinkDirtyDelta: 1/);
  assert.match(pageSource, /sinkDirtyDelta: completed \? -1 : 0/);
  assert.match(pageSource, /data-dirty-count=\{sinkDirtyPlateCount\}/);
  assert.match(
    pageSource,
    /洗碗池内有 \$\{sinkDirtyPlateCount\} 个脏盘，一次清洗 1 个/,
  );
  assert.match(pageSource, /<small>DIRTY<\/small>/);
  assert.match(pageSource, /className="wash-progress"/);
  assert.match(pageSource, /role="progressbar"/);
  assert.match(pageSource, /aria-valuenow=\{washingCompletedSteps\}/);
  assert.match(
    pageSource,
    /STEP \{washingCompletedSteps\} \/{" "}/,
  );
  assert.match(
    pageSource,
    /Sink capacity violated: exactly one dirty plate at a time/,
  );
  assert.match(cssSource, /\.sink-occupancy \{/);
  assert.match(cssSource, /\.pictogram-sink i:nth-child\(1\) \{/);
  assert.match(cssSource, /\.pictogram-sink i:nth-child\(2\) \{/);
  assert.match(cssSource, /\.pictogram-sink i:nth-child\(3\) \{/);
  assert.doesNotMatch(cssSource, /\.sink-occupancy\.empty \{/);
});

test("starts the rule-audited agent replay automatically at inspection speed", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(pageSource, /const \[playing, setPlaying\] = useState\(true\)/);
  assert.match(pageSource, /const \[speed, setSpeed\] = useState\(1\)/);
  assert.match(pageSource, /advanceMotionState\(/);
});

test("one to four scripted agents complete legal lockstep episodes", async () => {
  const {
    maps,
    orthogonalizePath,
    createMotionState,
    advanceMotionState,
    pointKey,
    countPhysicalPlates,
    countPhysicalExtinguishers,
  } = await loadPureSimulation();
  const map = maps[0];
  const paths = map.paths.map(orthogonalizePath);
  const blockedCells = new Set([
    ...map.counters.map(pointKey),
    ...map.stations.map((station) => pointKey([station.x, station.y])),
  ]);
  const deliveries = [];

  for (let agentCount = 1; agentCount <= 4; agentCount += 1) {
    let state = createMotionState(paths);
    for (let step = 0; step < 429; step += 1) {
      const previousPositions = state.agents.map((agent) => agent.position);
      state = advanceMotionState(
        state,
        paths,
        agentCount,
        blockedCells,
        map,
      );
      const positions = state.agents
        .slice(0, agentCount)
        .map((agent) => pointKey(agent.position));
      assert.equal(new Set(positions).size, agentCount);
      state.agents.slice(0, agentCount).forEach((agent, index) => {
        const moved =
          Math.abs(agent.position[0] - previousPositions[index][0]) +
          Math.abs(agent.position[1] - previousPositions[index][1]);
        assert.ok(moved <= 1, "movement must be cardinal lockstep");
        assert.notEqual(agent.carrying, "cooked-beef");
      });
      assert.equal(countPhysicalPlates(state), 4);
      assert.equal(countPhysicalExtinguishers(state), 1);
    }
    deliveries.push(state.servedOrders);
  }

  assert.deepEqual(deliveries, [4, 6, 12, 10]);
});

test("recovers a natural single-agent fire with one physical extinguisher", async () => {
  const {
    maps,
    orthogonalizePath,
    createMotionState,
    advanceMotionState,
    pointKey,
    countPhysicalPlates,
    countPhysicalExtinguishers,
  } = await loadPureSimulation();
  const map = maps[0];
  const paths = map.paths.map(orthogonalizePath);
  const blockedCells = new Set([
    ...map.counters.map(pointKey),
    ...map.stations.map((station) => pointKey([station.x, station.y])),
  ]);
  let state = createMotionState(paths);

  for (let step = 0; step < 429; step += 1) {
    state = advanceMotionState(
      state,
      paths,
      1,
      blockedCells,
      map,
    );
    assert.equal(countPhysicalPlates(state), 4);
    assert.equal(countPhysicalExtinguishers(state), 1);
  }

  assert.ok(state.firesStarted >= 1);
  assert.equal(state.firesExtinguished, state.firesStarted);
  assert.notEqual(state.grillFood, "burnt-beef");
});

test("washing returns one clean plate to hand before rack stacking", async () => {
  const {
    maps,
    orthogonalizePath,
    createMotionState,
    advanceMotionState,
    pointKey,
  } = await loadPureSimulation();
  const map = maps[0];
  const paths = map.paths.map(orthogonalizePath);
  const blockedCells = new Set([
    ...map.counters.map(pointKey),
    ...map.stations.map((station) => pointKey([station.x, station.y])),
  ]);
  let state = createMotionState(paths);
  let sawWashedPlateInHand = false;
  let sawRackReturn = false;

  for (let step = 0; step < 429; step += 1) {
    const previous = state;
    state = advanceMotionState(
      state,
      paths,
      3,
      blockedCells,
      map,
    );
    assert.ok(state.cleanPlatesAtRack >= 0);
    assert.ok(state.cleanPlatesAtRack <= 4);
    if (state.washedPlates > previous.washedPlates) {
      sawWashedPlateInHand = state.agents
        .slice(0, 3)
        .some((agent) => agent.carrying === "clean-plate");
    }
    if (state.cleanPlatesAtRack > previous.cleanPlatesAtRack) {
      sawRackReturn = true;
    }
  }

  assert.ok(sawWashedPlateInHand);
  assert.ok(sawRackReturn);
});

test("keeps unrelated agents moving while a teammate interacts", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(pageSource, /const targetGroups = new Map<string, number\[\]>/);
  assert.match(pageSource, /仅冲突主体等待/);
  assert.doesNotMatch(pageSource, /if \(duplicateTarget \|\| headOnSwap\)/);
});

test("labels pickup and sustained processing as work", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(pageSource, /if \(\/取\/\.test\(label\)\) return "WORK"/);
  assert.match(pageSource, /if \(\/放\|归还\|送餐\/\.test\(label\)\) return "DROP"/);
  assert.match(
    pageSource,
    /motion\.workKind === "washing" \|\| motion\.lastAction === "PROCESS"/,
  );
  assert.doesNotMatch(
    pageSource,
    /motion\.blockedKind === "interaction"\s*\?\s*"WORK"/,
  );
});

test("adjacent context actions do not require a facing turn", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(
    pageSource,
    /if \(!isCardinalNeighbor\(current\.position, target\)\) return null;\s*return onInteract\(\)/,
  );
  assert.doesNotMatch(pageSource, /isFacingTarget/);
  assert.doesNotMatch(pageSource, /turnToward/);
});

test("keeps the simulation floor free of decorative circles and zone watermarks", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );
  const cssSource = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );

  assert.doesNotMatch(pageSource, /kitchen-ambient/);
  assert.doesNotMatch(pageSource, /zone-label/);
  assert.doesNotMatch(cssSource, /\.kitchen-ambient/);
  assert.doesNotMatch(cssSource, /\.zone-label/);
});

test("combines held plated partials directly at dispensers and the grill", async () => {
  const pageSource = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(
    pageSource,
    /"plate-lettuce\+bun": "plate-bun-lettuce"/,
  );
  assert.match(
    pageSource,
    /"plate-bun-lettuce\+cooked-beef": "plated-burger"/,
  );
  assert.match(pageSource, /const directSources:/);
  assert.match(pageSource, /食材源可直接加入手持餐盘/);
  assert.match(pageSource, /STAY · 等待牛肉煎熟/);
  assert.match(pageSource, /从煎台直接取熟牛肉并完成/);
  assert.match(pageSource, /\{ grillAction: "take-cooked" \}/);
  assert.match(pageSource, /Cooked beef requires a held plate/);
  assert.doesNotMatch(
    pageSource,
    /current\.carrying === "cooked-beef" &&\s*worldState\.counterObjects\[4\]/,
  );
});
