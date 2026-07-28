"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";

type StationKind =
  | "bun"
  | "lettuce"
  | "beef"
  | "grill"
  | "plate"
  | "sink"
  | "serve"
  | "return"
  | "extinguisher"
  | "trash";

type GridPoint = [number, number];

type FoodKind =
  | "bun"
  | "lettuce"
  | "raw-beef"
  | "cooked-beef"
  | "burnt-beef"
  | "clean-plate"
  | "plate-bun"
  | "plate-lettuce"
  | "plate-cooked-beef"
  | "plate-bun-lettuce"
  | "plate-bun-cooked-beef"
  | "plate-lettuce-cooked-beef"
  | "dirty-plate"
  | "plated-burger"
  | "extinguisher";

type KitchenMap = {
  id: string;
  cols: number;
  rows: number;
  shortName: string;
  title: string;
  subtitle: string;
  accent: string;
  bottleneck: string;
  stations: Array<{ kind: StationKind; x: number; y: number }>;
  counters: GridPoint[];
  paths: GridPoint[][];
  results: number[];
  rewards: number[];
  coordination: number[];
  eventLabel: string;
};

const stationMeta: Record<StationKind, { icon: string; label: string }> = {
  bun: { icon: "◒", label: "汉堡胚" },
  lettuce: { icon: "≋", label: "生菜" },
  beef: { icon: "●", label: "生牛肉" },
  grill: { icon: "▥", label: "煎台" },
  plate: { icon: "○", label: "净盘" },
  sink: { icon: "≈", label: "洗碗池" },
  serve: { icon: "⌁", label: "送餐口" },
  return: { icon: "↩", label: "出餐口 · 脏盘回收" },
  extinguisher: { icon: "灭", label: "灭火器" },
  trash: { icon: "⌫", label: "垃圾桶" },
};

const maps: KitchenMap[] = [
  {
    id: "cramped",
    cols: 8,
    rows: 8,
    shortName: "狭窄厨房",
    title: "Cramped Galley",
    subtitle: "参考官方紧凑厨房 · 外围工作台 + 中央岛台",
    accent: "#ff7849",
    bottleneck: "双单格瓶颈 + 实体岛台",
    results: [3, 7, 10, 10],
    rewards: [47, 126, 181, 169],
    coordination: [42, 72, 87, 82],
    eventLabel: "单格交汇口",
    stations: [
      { kind: "bun", x: 0, y: 0 },
      { kind: "lettuce", x: 1, y: 0 },
      { kind: "beef", x: 3, y: 0 },
      { kind: "grill", x: 5, y: 0 },
      { kind: "extinguisher", x: 6, y: 0 },
      { kind: "trash", x: 7, y: 0 },
      { kind: "sink", x: 0, y: 7 },
      { kind: "plate", x: 1, y: 7 },
      { kind: "return", x: 6, y: 7 },
      { kind: "serve", x: 7, y: 7 },
    ],
    counters: [
      [0, 3],
      [1, 3],
      [3, 3],
      [4, 3],
      [5, 3],
      [7, 3],
      [3, 4],
      [4, 4],
      [2, 0],
      [4, 0],
      [2, 7],
      [3, 7],
      [4, 7],
      [5, 7],
    ],
    paths: [
      [
        [0, 1],
        [6, 1],
        [6, 4],
        [7, 4],
        [7, 6],
        [0, 6],
        [0, 4],
        [2, 4],
        [2, 2],
        [4, 2],
        [0, 2],
      ],
      [
        [3, 1],
        [6, 1],
        [7, 1],
        [6, 1],
        [6, 4],
        [7, 4],
        [7, 6],
        [0, 6],
        [0, 4],
        [2, 4],
        [2, 2],
        [0, 2],
        [0, 1],
      ],
      [
        [1, 6],
        [0, 6],
        [0, 4],
        [2, 4],
        [2, 2],
        [4, 2],
        [0, 2],
        [0, 1],
        [6, 1],
        [6, 4],
        [7, 4],
        [7, 6],
      ],
      [
        [7, 6],
        [0, 6],
        [0, 4],
        [2, 4],
        [2, 2],
        [0, 2],
        [0, 1],
        [6, 1],
        [6, 4],
        [7, 4],
      ],
    ],
  },
  {
    id: "split",
    cols: 12,
    rows: 8,
    shortName: "分区交接",
    title: "Split Service",
    subtitle: "经典 Forced Coordination 改造 · 强制交接",
    accent: "#7c8cff",
    bottleneck: "唯一传菜窗口",
    results: [2, 6, 11, 14],
    rewards: [31, 110, 214, 274],
    coordination: [38, 75, 91, 94],
    eventLabel: "传菜窗口",
    stations: [
      { kind: "bun", x: 0, y: 0 },
      { kind: "lettuce", x: 3, y: 0 },
      { kind: "beef", x: 0, y: 7 },
      { kind: "grill", x: 3, y: 7 },
      { kind: "extinguisher", x: 5, y: 7 },
      { kind: "plate", x: 8, y: 0 },
      { kind: "sink", x: 8, y: 7 },
      { kind: "serve", x: 11, y: 7 },
    ],
    counters: [
      [5, 0],
      [6, 0],
      [5, 1],
      [6, 1],
      [5, 2],
      [6, 2],
      [5, 4],
      [6, 4],
      [5, 5],
      [6, 5],
      [5, 6],
      [6, 6],
      [5, 7],
      [6, 7],
      [11, 0],
    ],
    paths: [
      [
        [1, 1],
        [2, 1],
        [3, 2],
        [4, 2],
        [4, 3],
        [3, 4],
        [2, 5],
        [1, 6],
      ],
      [
        [1, 6],
        [2, 6],
        [3, 6],
        [4, 5],
        [4, 3],
        [3, 3],
        [2, 4],
        [1, 4],
      ],
      [
        [7, 3],
        [8, 2],
        [9, 2],
        [10, 1],
        [10, 3],
        [9, 4],
        [8, 4],
        [7, 3],
      ],
      [
        [8, 6],
        [9, 6],
        [10, 6],
        [10, 5],
        [10, 4],
        [9, 4],
        [8, 5],
        [7, 5],
      ],
    ],
  },
  {
    id: "ring",
    cols: 12,
    rows: 8,
    shortName: "环形流水线",
    title: "Coordination Ring",
    subtitle: "经典 Coordination Ring 改造 · 流水线协同",
    accent: "#19b88a",
    bottleneck: "单向环路 + 对向避让",
    results: [2, 5, 9, 11],
    rewards: [30, 91, 170, 191],
    coordination: [36, 68, 85, 88],
    eventLabel: "东侧转角",
    stations: [
      { kind: "bun", x: 0, y: 1 },
      { kind: "lettuce", x: 0, y: 5 },
      { kind: "beef", x: 3, y: 7 },
      { kind: "grill", x: 8, y: 7 },
      { kind: "extinguisher", x: 11, y: 6 },
      { kind: "plate", x: 11, y: 3 },
      { kind: "serve", x: 4, y: 0 },
      { kind: "sink", x: 11, y: 1 },
    ],
    counters: [
      [3, 2],
      [4, 2],
      [5, 2],
      [6, 2],
      [7, 2],
      [8, 2],
      [3, 3],
      [8, 3],
      [3, 4],
      [8, 4],
      [3, 5],
      [4, 5],
      [5, 5],
      [6, 5],
      [7, 5],
      [8, 5],
      [8, 0],
    ],
    paths: [
      [
        [1, 1],
        [2, 1],
        [2, 2],
        [2, 4],
        [2, 6],
        [4, 6],
        [5, 6],
        [5, 5],
      ],
      [
        [4, 1],
        [6, 1],
        [8, 1],
        [9, 1],
        [9, 3],
        [9, 5],
        [8, 6],
        [7, 6],
      ],
      [
        [4, 6],
        [6, 6],
        [8, 6],
        [9, 5],
        [9, 3],
        [9, 1],
        [7, 1],
        [6, 1],
      ],
      [
        [1, 4],
        [1, 2],
        [2, 1],
        [4, 1],
        [6, 1],
        [8, 1],
        [9, 2],
        [9, 4],
      ],
    ],
  },
];

const teamProfiles = [
  {
    name: "MISO-01",
    role: "餐盘组装 / 送餐",
    color: "#ff7849",
  },
  {
    name: "MISO-02",
    role: "煎制",
    color: "#7c8cff",
  },
  {
    name: "MISO-03",
    role: "退盘 / 清洗",
    color: "#19b88a",
  },
  {
    name: "MISO-04",
    role: "消防安全",
    color: "#e4a83d",
  },
];

const benchmarks = [
  {
    agents: 1,
    orders: 3.0,
    throughput: 1.0,
    latency: 54,
    idle: 32,
    collision: 0.2,
    specialization: 38,
    handoff: 0,
  },
  {
    agents: 2,
    orders: 7.0,
    throughput: 2.33,
    latency: 36,
    idle: 18,
    collision: 1.1,
    specialization: 68,
    handoff: 79,
  },
  {
    agents: 3,
    orders: 10.0,
    throughput: 3.33,
    latency: 29,
    idle: 9,
    collision: 2.5,
    specialization: 86,
    handoff: 91,
  },
  {
    agents: 4,
    orders: 10.0,
    throughput: 3.33,
    latency: 32,
    idle: 14,
    collision: 5.8,
    specialization: 81,
    handoff: 88,
  },
];

const rewardRows = [
  ["正确出餐", "+20.0", "团队"],
  ["势函数差分", "γΦ(s′) − Φ(s)", "共享"],
  ["首次烧糊 / 起火", "−5.0", "团队"],
  ["碰撞 / 阻塞", "−0.05", "团队"],
  ["每步时间成本", "0.0", "固定时长内由吞吐目标隐式约束"],
  ["丢弃食材 / 倒空盘", "0.0", "无事件奖励"],
  ["非法 / 重复交互", "0.0", "无事件"],
];

const ppoTrainingProgress = [
  {
    steps: 8_192,
    deliveriesPerEpisode: 1.0,
    dpm: 0.333,
    phase: "固定开局巩固",
    protocol: "固定位置",
  },
  {
    steps: 131_072,
    deliveriesPerEpisode: 1.0,
    dpm: 0.333,
    phase: "固定开局巩固",
    protocol: "固定位置",
  },
  {
    steps: 262_144,
    deliveriesPerEpisode: 3.0,
    dpm: 0.999,
    phase: "固定开局巩固",
    protocol: "固定位置",
  },
  {
    steps: 270_336,
    deliveriesPerEpisode: 2.7,
    dpm: 0.899,
    phase: "任意位置泛化",
    protocol: "随机位置",
  },
  {
    steps: 393_216,
    deliveriesPerEpisode: 3.6,
    dpm: 1.199,
    phase: "任意位置泛化",
    protocol: "随机位置",
  },
  {
    steps: 524_288,
    deliveriesPerEpisode: 3.6,
    dpm: 1.199,
    phase: "任意位置泛化",
    protocol: "随机位置",
  },
  {
    steps: 532_480,
    deliveriesPerEpisode: 3.8,
    dpm: 1.265,
    phase: "脏盘可观测",
    protocol: "随机位置",
  },
  {
    steps: 655_360,
    deliveriesPerEpisode: 3.8,
    dpm: 1.265,
    phase: "脏盘可观测",
    protocol: "随机位置",
  },
  {
    steps: 786_432,
    deliveriesPerEpisode: 2.45,
    dpm: 0.816,
    phase: "脏盘可观测",
    protocol: "随机位置",
  },
  {
    steps: 794_624,
    deliveriesPerEpisode: 3.8,
    dpm: 1.265,
    phase: "困难开局强化",
    protocol: "回滚最佳点 · 随机位置",
  },
  {
    steps: 917_504,
    deliveriesPerEpisode: 3.8,
    dpm: 1.265,
    phase: "困难开局强化",
    protocol: "随机位置",
  },
  {
    steps: 1_048_576,
    deliveriesPerEpisode: 3.8,
    dpm: 1.265,
    phase: "困难开局强化",
    protocol: "随机位置",
  },
  {
    steps: 1_179_648,
    deliveriesPerEpisode: 5.2125,
    dpm: 1.7358,
    phase: "联合困难验收",
    protocol: "40 位置 × 4 场景",
  },
] as const;

const trainingPhases = [
  { start: 0, end: 262_144, label: "固定开局巩固" },
  { start: 262_144, end: 524_288, label: "任意位置泛化" },
  { start: 524_288, end: 786_432, label: "脏盘可观测" },
  { start: 786_432, end: 1_048_576, label: "困难开局强化" },
  { start: 1_048_576, end: 1_179_648, label: "联合验收" },
] as const;

function formatTime(seconds: number) {
  const safe = Math.max(0, seconds);
  return `${String(Math.floor(safe / 60)).padStart(2, "0")}:${String(
    Math.floor(safe % 60),
  ).padStart(2, "0")}`;
}

function formatScore(value: number) {
  const rounded = Math.round(value);
  return rounded < 0
    ? `−${String(Math.abs(rounded)).padStart(2, "0")}`
    : String(rounded).padStart(3, "0");
}

const foodLabels: Record<FoodKind, string> = {
  bun: "汉堡胚",
  lettuce: "生菜",
  "raw-beef": "生牛肉",
  "cooked-beef": "熟牛肉",
  "burnt-beef": "烧糊牛肉",
  "clean-plate": "净盘",
  "plate-bun": "餐盘 + 汉堡胚",
  "plate-lettuce": "餐盘 + 生菜",
  "plate-cooked-beef": "餐盘 + 熟牛肉",
  "plate-bun-lettuce": "餐盘 + 汉堡胚 + 生菜",
  "plate-bun-cooked-beef": "餐盘 + 汉堡胚 + 熟牛肉",
  "plate-lettuce-cooked-beef": "餐盘 + 生菜 + 熟牛肉",
  "dirty-plate": "脏盘",
  "plated-burger": "汉堡餐盘",
  extinguisher: "灭火器",
};

type PlatedLayer = "bottom-bun" | "lettuce" | "patty" | "top-bun";

const platedFoodLayers: Partial<
  Record<FoodKind, readonly PlatedLayer[]>
> = {
  "plate-bun": ["bottom-bun"],
  "plate-lettuce": ["lettuce"],
  "plate-cooked-beef": ["patty"],
  "plate-bun-lettuce": ["bottom-bun", "lettuce"],
  "plate-bun-cooked-beef": ["bottom-bun", "patty"],
  "plate-lettuce-cooked-beef": ["patty", "lettuce"],
  "plated-burger": ["bottom-bun", "patty", "lettuce", "top-bun"],
};

function FoodArt({
  kind,
  className = "",
}: {
  kind: FoodKind;
  className?: string;
}) {
  const platedLayers = platedFoodLayers[kind];
  return (
    <span
      className={`food-art food-${kind} ${
        platedLayers ? "food-with-visible-plate" : ""
      } ${className}`}
      role="img"
      aria-label={foodLabels[kind]}
      data-plate-visible={platedLayers ? "true" : undefined}
    >
      {platedLayers ? (
        <>
          <span className="food-plate-base" aria-hidden="true" />
          {platedLayers.map((layer, index) => (
            <i
              className={`food-layer food-layer-${layer}`}
              key={`${layer}-${index}`}
              style={
                { "--layer-offset": `${index * 5}px` } as CSSProperties
              }
            />
          ))}
        </>
      ) : (
        <>
          <i />
          <i />
          <i />
          <i />
        </>
      )}
    </span>
  );
}

function signedPercent(value: number) {
  const rounded = Math.round(value);
  return `${rounded > 0 ? "+" : ""}${rounded}%`;
}

function orthogonalizePath(waypoints: GridPoint[]) {
  if (waypoints.length === 0) return [];

  const path: GridPoint[] = [[...waypoints[0]]];
  const appendAxisAlignedSteps = ([targetX, targetY]: GridPoint) => {
    let [x, y] = path[path.length - 1];

    while (x !== targetX) {
      x += Math.sign(targetX - x);
      path.push([x, y]);
    }
    while (y !== targetY) {
      y += Math.sign(targetY - y);
      path.push([x, y]);
    }
  };

  for (let index = 1; index <= waypoints.length; index += 1) {
    appendAxisAlignedSteps(waypoints[index % waypoints.length]);
  }

  path.pop();
  return path;
}

type Facing = "north" | "south" | "east" | "west";

const facingArrows: Record<Facing, string> = {
  north: "↑",
  south: "↓",
  east: "→",
  west: "←",
};
const cardinalSteps: readonly GridPoint[] = [
  [0, -1],
  [0, 1],
  [1, 0],
  [-1, 0],
];

type AgentMotion = {
  position: GridPoint;
  cursor: number;
  orientation: Facing;
  lastAction:
    | "↑"
    | "↓"
    | "←"
    | "→"
    | "STAY"
    | "PICK/DROP"
    | "PROCESS";
  carrying: FoodKind | null;
  interactionCount: number;
  interactionLabel: string | null;
  workKind: "washing" | null;
  workTicksRemaining: number;
  workTotalTicks: number;
  pendingCarry: FoodKind | null;
  blocked: boolean;
  blockedReason: string | null;
  blockedKind: "collision" | "interaction" | null;
};

type MotionState = {
  agents: AgentMotion[];
  step: number;
  avoidedCollisions: number;
  dirtyPlatesAtReturn: number;
  dirtyPlatesInSink: number;
  pendingPlateReturns: number[];
  cleanPlatesAtRack: number;
  washedPlates: number;
  counterObjects: Partial<Record<number, FoodKind>>;
  discardedItems: number;
  firesStarted: number;
  firesExtinguished: number;
  extinguisherAtStation: boolean;
  grillFood: "raw-beef" | "cooked-beef" | "burnt-beef" | null;
  grillCookTicksRemaining: number;
  grillBurnTicksRemaining: number;
  servedOrders: number;
};

type PolicyReplayFrame = {
  state: MotionState;
  actionIndex: number;
  action: AgentMotion["lastAction"];
  reward: number;
  cumulativeReward: number;
  cumulativeSparseReward: number;
  events: string[];
};

type PolicyReplay = {
  schema: "nexus.burger.ppo-policy-replay.v1";
  algorithm: "PPO";
  execution: string;
  checkpoint: string;
  checkpointLabel: string;
  scenarioId: PolicyScenarioId;
  scenarioLabel: string;
  startStage: string;
  randomizedStart: boolean;
  curriculumStage: string;
  controlStepSeconds: number;
  validatedEpisodes: number;
  validatedSuccesses: number;
  deliveries: number;
  washedPlates: number;
  fires: number;
  firesExtinguished: number;
  deliveryStep: number | null;
  totalReward: number;
  sparseReward: number;
  frames: PolicyReplayFrame[];
};

type PolicyScenarioId = "standard" | "fire" | "dirty";

const policyScenarioOptions: {
  id: PolicyScenarioId;
  source: string;
  index: string;
  label: string;
  description: string;
}[] = [
  {
    id: "standard",
    source: "/ppo-policy-replay-standard.json",
    index: "01",
    label: "标准生产",
    description: "随机位置 · 完整出餐循环",
  },
  {
    id: "fire",
    source: "/ppo-policy-replay-fire.json",
    index: "02",
    label: "灭火恢复",
    description: "灶台起火 · 扑灭后恢复出餐",
  },
  {
    id: "dirty",
    source: "/ppo-policy-replay-dirty.json",
    index: "03",
    label: "脏盘回收",
    description: "手持脏盘 · 洗净后恢复生产",
  },
];

const WASH_DURATION_STEPS = 10;
const PLATE_RETURN_DELAY_STEPS = 12;
const CONTROL_STEP_SECONDS = 0.42;
const GRILL_COOK_STEPS = 24;
const GRILL_BURN_STEPS = 16;
const FIRE_STARTED_PENALTY = -5;
const BASELINE_RECOVERED_FAULTS = 2;
const EPISODE_STEPS = 429;
const TOTAL_PHYSICAL_PLATES = 4;
const plateBearingItems = new Set<FoodKind>([
  "clean-plate",
  "plate-bun",
  "plate-lettuce",
  "plate-cooked-beef",
  "plate-bun-lettuce",
  "plate-bun-cooked-beef",
  "plate-lettuce-cooked-beef",
  "dirty-plate",
  "plated-burger",
]);
const trashDiscardableItems = new Set<FoodKind>([
  "bun",
  "lettuce",
  "raw-beef",
  "cooked-beef",
  "burnt-beef",
  "plate-bun",
  "plate-lettuce",
  "plate-cooked-beef",
  "plate-bun-lettuce",
  "plate-bun-cooked-beef",
  "plate-lettuce-cooked-beef",
  "plated-burger",
]);
const loadedPlateItems = new Set<FoodKind>([
  "plate-bun",
  "plate-lettuce",
  "plate-cooked-beef",
  "plate-bun-lettuce",
  "plate-bun-cooked-beef",
  "plate-lettuce-cooked-beef",
  "plated-burger",
]);
type BurgerIngredient = "bun" | "lettuce" | "cooked-beef";
const directPlateCombinations: Record<string, FoodKind> = {
  "clean-plate+bun": "plate-bun",
  "clean-plate+lettuce": "plate-lettuce",
  "clean-plate+cooked-beef": "plate-cooked-beef",
  "plate-bun+lettuce": "plate-bun-lettuce",
  "plate-bun+cooked-beef": "plate-bun-cooked-beef",
  "plate-lettuce+bun": "plate-bun-lettuce",
  "plate-lettuce+cooked-beef": "plate-lettuce-cooked-beef",
  "plate-cooked-beef+bun": "plate-bun-cooked-beef",
  "plate-cooked-beef+lettuce": "plate-lettuce-cooked-beef",
  "plate-bun-lettuce+cooked-beef": "plated-burger",
  "plate-bun-cooked-beef+lettuce": "plated-burger",
  "plate-lettuce-cooked-beef+bun": "plated-burger",
};

function combinePlateWithIngredient(
  plate: FoodKind | null,
  ingredient: BurgerIngredient,
) {
  if (plate === null) return null;
  return directPlateCombinations[`${plate}+${ingredient}`] ?? null;
}

type WorldEffect = {
  counterMutation?: {
    counterIndex: number;
    from: FoodKind | null;
    to: FoodKind | null;
  };
  cleanPlateDelta?: number;
  extinguisherStationDelta?: -1 | 1;
  grillAction?: "place-raw" | "take-cooked" | "extinguish";
  servedDelta?: number;
  discardedDelta?: number;
  schedulePlateReturn?: boolean;
};

type InteractionResolution = {
  agent: AgentMotion;
  label: string | null;
  dishDelta: number;
  sinkDirtyDelta: number;
  washedDelta: number;
  actionKind: "interaction" | null;
  worldEffect?: WorldEffect;
};

function pointKey([x, y]: GridPoint) {
  return `${x},${y}`;
}

function countPhysicalPlates(state: MotionState) {
  return (
    state.cleanPlatesAtRack +
    state.dirtyPlatesAtReturn +
    state.dirtyPlatesInSink +
    state.pendingPlateReturns.length +
    Object.values(state.counterObjects ?? {}).reduce(
      (total, item) =>
        total + Number(item !== undefined && plateBearingItems.has(item)),
      0,
    ) +
    state.agents.reduce(
      (total, agent) =>
        total +
        Number(
          agent.carrying !== null &&
            plateBearingItems.has(agent.carrying),
        ),
      0,
    )
  );
}

function countPhysicalExtinguishers(state: MotionState) {
  return (
    Number(state.extinguisherAtStation) +
    Object.values(state.counterObjects ?? {}).filter(
      (item) => item === "extinguisher",
    ).length +
    state.agents.filter(
      (agent) => agent.carrying === "extinguisher",
    ).length
  );
}

function createMotionState(paths: GridPoint[][]): MotionState {
  return {
    agents: paths.map((path, index) => ({
      position: [...path[0]],
      cursor: 0,
      orientation: index < 2 ? "north" : "south",
      lastAction: "STAY",
      carrying: null,
      interactionCount: 0,
      interactionLabel: null,
      workKind: null,
      workTicksRemaining: 0,
      workTotalTicks: 0,
      pendingCarry: null,
      blocked: false,
      blockedReason: null,
      blockedKind: null,
    })),
    step: 0,
    avoidedCollisions: 0,
    dirtyPlatesAtReturn: 0,
    dirtyPlatesInSink: 0,
    pendingPlateReturns: [],
    cleanPlatesAtRack: 4,
    washedPlates: 0,
    counterObjects: {},
    discardedItems: 0,
    firesStarted: 0,
    firesExtinguished: 0,
    extinguisherAtStation: true,
    grillFood: null,
    grillCookTicksRemaining: 0,
    grillBurnTicksRemaining: 0,
    servedOrders: 0,
  };
}

function isCardinalNeighbor(position: GridPoint, target: GridPoint) {
  return (
    Math.abs(position[0] - target[0]) +
      Math.abs(position[1] - target[1]) ===
    1
  );
}

function roleIndices(activeCount: number) {
  return {
    assembler: 0,
    cook: activeCount >= 2 ? 1 : 0,
    dish: activeCount >= 3 ? 2 : 0,
    prep: null,
    safety: activeCount >= 4 ? 3 : null,
  };
}

function targetForAgent(
  agent: AgentMotion,
  index: number,
  activeCount: number,
  map: KitchenMap,
  state: MotionState,
): GridPoint | null {
  const roles = roleIndices(activeCount);
  const station = (kind: StationKind): GridPoint => {
    const match = map.stations.find((item) => item.kind === kind);
    return match ? [match.x, match.y] : [-100, -100];
  };

  if (agent.workKind === "washing" && agent.workTicksRemaining > 0) {
    return station("sink");
  }
  const fireResponder = roles.safety ?? roles.cook;
  if (
    index === fireResponder &&
    state.grillFood === "burnt-beef"
  ) {
    if (agent.carrying === "extinguisher") return station("grill");
    if (agent.carrying === "clean-plate") return station("plate");
    if (agent.carrying !== null) {
      const emptyCounterIndex = map.counters.findIndex(
        (_, counterIndex) =>
          state.counterObjects[counterIndex] === undefined,
      );
      if (emptyCounterIndex >= 0) {
        return map.counters[emptyCounterIndex];
      }
      if (trashDiscardableItems.has(agent.carrying)) {
        return station("trash");
      }
      return null;
    }
    return state.extinguisherAtStation
      ? station("extinguisher")
      : null;
  }
  if (index === roles.safety) {
    if (agent.carrying === "extinguisher") {
      return map.counters[5];
    }
    if (agent.carrying === null && state.extinguisherAtStation) {
      return station("extinguisher");
    }
  }
  if (index === roles.dish) {
    if (agent.carrying === "dirty-plate") return station("sink");
    if (
      agent.carrying === "clean-plate" &&
      roles.dish !== roles.assembler
    ) {
      return station("plate");
    }
    if (agent.carrying === null && state.dirtyPlatesAtReturn > 0) {
      return station("return");
    }
    if (
      agent.carrying === null &&
      roles.dish !== roles.assembler
    ) {
      return station("return");
    }
  }

  if (index === roles.cook) {
    if (agent.carrying === "extinguisher") return station("extinguisher");
    if (agent.carrying === "raw-beef") return station("grill");
    if (agent.carrying === null && state.grillFood === null) {
      return station("beef");
    }
    if (
      agent.carrying === null &&
      state.grillFood !== null &&
      index !== roles.assembler
    ) {
      return station("beef");
    }
  }

  if (index === roles.assembler) {
    if (agent.carrying === "plated-burger") return station("serve");
    if (agent.carrying === null) return station("plate");
    if (plateBearingItems.has(agent.carrying)) {
      const layers = platedFoodLayers[agent.carrying] ?? [];
      if (!layers.includes("bottom-bun")) {
        return state.counterObjects[2] === "bun"
          ? map.counters[2]
          : station("bun");
      }
      if (!layers.includes("lettuce")) {
        return state.counterObjects[3] === "lettuce"
          ? map.counters[3]
          : station("lettuce");
      }
      if (!layers.includes("patty")) {
        return state.grillFood === "raw-beef" ||
          state.grillFood === "cooked-beef"
          ? station("grill")
          : station("plate");
      }
    }
  }

  if (roles.prep === index) {
    const desiredIngredient =
      Math.floor(agent.interactionCount / 2) % 2 === 0 ? "bun" : "lettuce";
    if (agent.carrying === null) return station(desiredIngredient);
    if (agent.carrying === "bun") return map.counters[2];
    if (agent.carrying === "lettuce") return map.counters[3];
  }

  return null;
}

function nextFloorStep(
  start: GridPoint,
  target: GridPoint | null,
  map: KitchenMap,
  blockedCells: Set<string>,
  occupiedCells: Set<string>,
): GridPoint {
  if (target === null || isCardinalNeighbor(start, target)) return start;
  const goals = new Set(
    cardinalSteps
      .map(([dx, dy]) => [target[0] + dx, target[1] + dy] as GridPoint)
      .filter(
        ([x, y]) =>
          x >= 0 &&
          y >= 0 &&
          x < map.cols &&
          y < map.rows &&
          !blockedCells.has(pointKey([x, y])),
      )
      .map(pointKey),
  );
  const queue: Array<{ position: GridPoint; first: GridPoint | null }> = [
    { position: start, first: null },
  ];
  const visited = new Set([pointKey(start)]);
  while (queue.length > 0) {
    const current = queue.shift()!;
    if (goals.has(pointKey(current.position))) {
      return current.first ?? start;
    }
    for (const [dx, dy] of cardinalSteps) {
      const next: GridPoint = [
        current.position[0] + dx,
        current.position[1] + dy,
      ];
      const key = pointKey(next);
      if (
        visited.has(key) ||
        next[0] < 0 ||
        next[1] < 0 ||
        next[0] >= map.cols ||
        next[1] >= map.rows ||
        blockedCells.has(key) ||
        (key !== pointKey(start) && occupiedCells.has(key))
      ) {
        continue;
      }
      visited.add(key);
      queue.push({
        position: next,
        first: current.first ?? next,
      });
    }
  }
  return start;
}

function facingToward(position: GridPoint, target: GridPoint): Facing | null {
  const dx = target[0] - position[0];
  const dy = target[1] - position[1];

  if (dx === 1 && dy === 0) return "east";
  if (dx === -1 && dy === 0) return "west";
  if (dx === 0 && dy === 1) return "south";
  if (dx === 0 && dy === -1) return "north";
  return null;
}

function resolveAgentInteraction(
  agent: AgentMotion,
  index: number,
  activeCount: number,
  map: KitchenMap,
  plannedTarget: GridPoint | null,
  dirtyPlatesAtReturn: number,
  worldState: Pick<
    MotionState,
    | "counterObjects"
    | "cleanPlatesAtRack"
    | "grillFood"
    | "dirtyPlatesInSink"
    | "extinguisherAtStation"
  >,
): InteractionResolution {
  const roles = roleIndices(activeCount);
  const grillNeedsRecovery = worldState.grillFood === "burnt-beef";
  const fireResponder = roles.safety ?? roles.cook;

  if (agent.workKind === "washing" && agent.workTicksRemaining > 0) {
    const workTicksRemaining = agent.workTicksRemaining - 1;
    const completed = workTicksRemaining === 0;
    const completedSteps = agent.workTotalTicks - workTicksRemaining;
    const label = completed
      ? "洗盘完成 · 净盘回到手中"
      : `洗盘中 · ${completedSteps}/${agent.workTotalTicks}`;

    return {
      agent: {
        ...agent,
        carrying: completed ? agent.pendingCarry : agent.carrying,
        interactionLabel: label,
        lastAction: "PROCESS" as const,
        workKind: completed ? null : agent.workKind,
        workTicksRemaining,
        workTotalTicks: completed ? 0 : agent.workTotalTicks,
        pendingCarry: completed ? null : agent.pendingCarry,
      },
      label,
      dishDelta: 0,
      sinkDirtyDelta: completed ? -1 : 0,
      washedDelta: completed ? 1 : 0,
      actionKind: "interaction" as const,
    };
  }

  const current = { ...agent, interactionLabel: null };
  const stationPoint = (kind: StationKind): GridPoint => {
    const station = map.stations.find((item) => item.kind === kind);
    return station ? [station.x, station.y] : [-100, -100];
  };
  const interact = (
    carrying: FoodKind | null,
    label: string,
    dishDelta = 0,
    worldEffect?: WorldEffect,
    controlAction: "PICK/DROP" | "PROCESS" = "PICK/DROP",
  ) => ({
    agent: {
      ...current,
      carrying,
      interactionCount: current.interactionCount + 1,
      interactionLabel: label,
      lastAction: controlAction,
    },
    label,
    dishDelta,
    sinkDirtyDelta: 0,
    washedDelta: 0,
    actionKind: "interaction" as const,
    worldEffect,
  });
  const actAtTarget = (
    target: GridPoint,
    onInteract: () => ReturnType<typeof interact>,
  ) => {
    if (
      plannedTarget === null ||
      pointKey(plannedTarget) !== pointKey(target)
    ) {
      return null;
    }
    if (!isCardinalNeighbor(current.position, target)) return null;
    return onInteract();
  };
  const actAtStation = (
    kind: StationKind,
    onInteract: () => ReturnType<typeof interact>,
  ) => actAtTarget(stationPoint(kind), onInteract);
  const actAtCounter = (
    counterIndex: number,
    onInteract: () => ReturnType<typeof interact>,
  ) => actAtTarget(map.counters[counterIndex], onInteract);
  const waitAtTarget = (target: GridPoint, label: string) => {
    if (
      plannedTarget === null ||
      pointKey(plannedTarget) !== pointKey(target)
    ) {
      return null;
    }
    if (!isCardinalNeighbor(current.position, target)) return null;
    return {
      agent: {
        ...current,
        interactionLabel: label,
        lastAction: "STAY" as const,
      },
      label,
      dishDelta: 0,
      sinkDirtyDelta: 0,
      washedDelta: 0,
      actionKind: "interaction" as const,
    };
  };
  const startWashing = () => ({
    agent: {
      ...current,
      carrying: null,
      interactionCount: current.interactionCount + 1,
      interactionLabel: `开始洗盘 · 0/${WASH_DURATION_STEPS}`,
      lastAction: "PICK/DROP" as const,
      workKind: "washing" as const,
      workTicksRemaining: WASH_DURATION_STEPS,
      workTotalTicks: WASH_DURATION_STEPS,
      pendingCarry: "clean-plate" as FoodKind,
    },
    label: `开始洗盘 · 0/${WASH_DURATION_STEPS}`,
    dishDelta: 0,
    sinkDirtyDelta: 1,
    washedDelta: 0,
    actionKind: "interaction" as const,
  });

  if (index === fireResponder && grillNeedsRecovery) {
    if (current.carrying === "clean-plate") {
      const action = actAtStation("plate", () =>
        interact(null, "PICK/DROP · 应急放回净盘", 0, {
          cleanPlateDelta: 1,
        }),
      );
      if (action) return action;
    }
    if (
      current.carrying !== null &&
      current.carrying !== "extinguisher"
    ) {
      const emptyCounterIndex = map.counters.findIndex(
        (_, counterIndex) =>
          worldState.counterObjects[counterIndex] === undefined,
      );
      if (emptyCounterIndex >= 0) {
        const heldItem = current.carrying;
        const action = actAtCounter(emptyCounterIndex, () =>
          interact(
            null,
            `PICK/DROP · 起火应急放下${foodLabels[heldItem]}`,
            0,
            {
              counterMutation: {
                counterIndex: emptyCounterIndex,
                from: null,
                to: heldItem,
              },
            },
          ),
        );
        if (action) return action;
      }
    }
    if (
      current.carrying === null &&
      worldState.extinguisherAtStation
    ) {
      const action = actAtStation("extinguisher", () =>
        interact("extinguisher", "PICK/DROP · 取唯一灭火器", 0, {
          extinguisherStationDelta: -1,
        }),
      );
      if (action) return action;
    }
    if (current.carrying === "extinguisher") {
      const action = actAtStation("grill", () =>
        interact(
          "extinguisher",
          "PROCESS · 煎台灭火",
          0,
          { grillAction: "extinguish" },
          "PROCESS",
        ),
      );
      if (action) return action;
    }
  }

  if (
    (index === roles.cook || index === roles.safety) &&
    grillNeedsRecovery &&
    current.carrying !== null &&
    trashDiscardableItems.has(current.carrying)
  ) {
    const discarded = current.carrying;
    const retainedPlate = loadedPlateItems.has(discarded);
    const action = actAtStation("trash", () =>
      interact(
        retainedPlate ? "clean-plate" : null,
        retainedPlate
          ? `PICK/DROP · 倒掉${foodLabels[discarded]}内容 · 保留净盘`
          : `PICK/DROP · 丢弃${foodLabels[discarded]}`,
        0,
        { discardedDelta: 1 },
      ),
    );
    if (action) return action;
  }

  if (index === roles.safety) {
    if (
      !grillNeedsRecovery &&
      current.carrying === null &&
      worldState.extinguisherAtStation
    ) {
      const action = actAtStation("extinguisher", () =>
        interact("extinguisher", "PICK/DROP · 取唯一灭火器", 0, {
          extinguisherStationDelta: -1,
        }),
      );
      if (action) return action;
    }
  }

  if (index === roles.assembler) {
    const directSources: Array<{
      kind: "bun" | "lettuce";
      ingredient: "bun" | "lettuce";
    }> = [
      { kind: "bun", ingredient: "bun" },
      { kind: "lettuce", ingredient: "lettuce" },
    ];
    for (const source of directSources) {
      const combined = combinePlateWithIngredient(
        current.carrying,
        source.ingredient,
      );
      if (combined === null) continue;
      const action = actAtStation(source.kind, () =>
        interact(
          combined,
          `PICK/DROP · ${foodLabels[source.ingredient]}直接加入${foodLabels[current.carrying!]}`,
        ),
      );
      if (action) return action;
    }

    const withCookedBeef = combinePlateWithIngredient(
      current.carrying,
      "cooked-beef",
    );
    if (
      withCookedBeef !== null &&
      worldState.grillFood === "raw-beef"
    ) {
      const action = waitAtTarget(
        stationPoint("grill"),
        "STAY · 等待牛肉煎熟",
      );
      if (action) return action;
    }
    if (
      withCookedBeef !== null &&
      worldState.grillFood === "cooked-beef"
    ) {
      const action = actAtStation("grill", () =>
        interact(
          withCookedBeef,
          `PICK/DROP · 从煎台直接取熟牛肉并完成${foodLabels[withCookedBeef]}`,
          0,
          { grillAction: "take-cooked" },
        ),
      );
      if (action) return action;
    }
  }

  if (index === roles.prep) {
    const desiredIngredient =
      Math.floor(current.interactionCount / 2) % 2 === 0 ? "bun" : "lettuce";
    if (current.carrying === null && desiredIngredient === "bun") {
      const action = actAtStation("bun", () =>
        interact("bun", "PICK/DROP · 取汉堡胚"),
      );
      if (action) return action;
    }
    if (current.carrying === null && desiredIngredient === "lettuce") {
      const action = actAtStation("lettuce", () =>
        interact("lettuce", "PICK/DROP · 取生菜"),
      );
      if (action) return action;
    }
    if (current.carrying === "bun" || current.carrying === "lettuce") {
      const isBun = current.carrying === "bun";
      const counterIndex = isBun ? 2 : 3;
      const targetIsFree = worldState.counterObjects[counterIndex] == null;
      const action = targetIsFree
        ? actAtCounter(counterIndex, () =>
            interact(
              null,
              `PICK/DROP · 放下${foodLabels[current.carrying!]}`,
              0,
              {
                counterMutation: {
                  counterIndex,
                  from: null,
                  to: isBun ? "bun" : "lettuce",
                },
              },
            ),
          )
        : null;
      if (action) return action;
    }
  }

  if (index === roles.cook) {
    if (!grillNeedsRecovery && current.carrying === "extinguisher") {
      const action = actAtStation("extinguisher", () =>
        interact(null, "PICK/DROP · 归还唯一灭火器", 0, {
          extinguisherStationDelta: 1,
        }),
      );
      if (action) return action;
    }
    if (
      !grillNeedsRecovery &&
      current.carrying === null &&
      worldState.grillFood === null
    ) {
      const action = actAtStation("beef", () =>
        interact("raw-beef", "PICK/DROP · 取生牛肉"),
      );
      if (action) return action;
    }
    if (current.carrying === "raw-beef" && worldState.grillFood === null) {
      const action = actAtStation("grill", () =>
        interact(null, "PICK/DROP · 生牛肉放上煎台", 0, {
          grillAction: "place-raw",
        }),
      );
      if (action) return action;
    }
  }

  if (index === roles.assembler) {
    if (
      current.carrying === null &&
      !(index === roles.dish && dirtyPlatesAtReturn > 0) &&
      worldState.cleanPlatesAtRack > 0
    ) {
      const action = actAtStation("plate", () =>
        interact("clean-plate", "PICK/DROP · 取净盘", 0, {
          cleanPlateDelta: -1,
        }),
      );
      if (action) return action;
    }
    if (
      current.carrying === "clean-plate" &&
      worldState.counterObjects[2] === "bun"
    ) {
      const action = actAtCounter(2, () =>
        interact("plate-bun", "PICK/DROP · 餐盘取汉堡胚", 0, {
          counterMutation: {
            counterIndex: 2,
            from: "bun",
            to: null,
          },
        }),
      );
      if (action) return action;
    }
    if (
      current.carrying === "plate-bun" &&
      worldState.counterObjects[3] === "lettuce"
    ) {
      const action = actAtCounter(3, () =>
        interact(
          "plate-bun-lettuce",
          "PICK/DROP · 餐盘取生菜",
          0,
          {
            counterMutation: {
              counterIndex: 3,
              from: "lettuce",
              to: null,
            },
          },
        ),
      );
      if (action) return action;
    }
    if (
      current.carrying === "plate-bun-lettuce" &&
      worldState.counterObjects[4] === "cooked-beef"
    ) {
      const action = actAtCounter(4, () =>
        interact(
          "plated-burger",
          "PICK/DROP · 餐盘取熟牛肉",
          0,
          {
            counterMutation: {
              counterIndex: 4,
              from: "cooked-beef",
              to: null,
            },
          },
        ),
      );
      if (action) return action;
    }
    if (current.carrying === "plated-burger") {
      const action = actAtStation("serve", () =>
        interact(null, "PICK/DROP · 送餐成功 · 餐盘进入延迟回收", 0, {
          servedDelta: 1,
          schedulePlateReturn: true,
        }),
      );
      if (action) return action;
    }
  }

  if (index === roles.dish) {
    if (current.carrying === null && dirtyPlatesAtReturn > 0) {
      const action = actAtStation("return", () =>
        interact("dirty-plate", "PICK/DROP · 从回收口取脏盘", -1),
      );
      if (action) return action;
    }
    if (
      current.carrying === "dirty-plate" &&
      worldState.dirtyPlatesInSink === 0
    ) {
      const action = actAtTarget(
        stationPoint("sink"),
        startWashing,
      );
      if (action) return action;
    }
    if (
      current.carrying === "clean-plate" &&
      roles.dish !== roles.assembler
    ) {
      const action = actAtStation("plate", () =>
        interact(null, "PICK/DROP · 放回净盘", 0, {
          cleanPlateDelta: 1,
        }),
      );
      if (action) return action;
    }
  }

  return {
    agent: current,
    label: null,
    dishDelta: 0,
    sinkDirtyDelta: 0,
    washedDelta: 0,
    actionKind: null,
  };
}

function advanceMotionState(
  state: MotionState,
  paths: GridPoint[][],
  activeCount: number,
  blockedCells: Set<string>,
  map: KitchenMap,
) {
  const activeIndices = Array.from({ length: activeCount }, (_, index) => index);
  const plateInventory = Number.isFinite(state.dirtyPlatesAtReturn)
    ? state.dirtyPlatesAtReturn
    : 0;
  const worldState = {
    counterObjects: { ...(state.counterObjects ?? {}) },
    cleanPlatesAtRack: Number.isFinite(state.cleanPlatesAtRack)
      ? state.cleanPlatesAtRack
      : 1,
    grillFood: state.grillFood ?? null,
    dirtyPlatesInSink: Number.isFinite(state.dirtyPlatesInSink)
      ? state.dirtyPlatesInSink
      : 0,
    extinguisherAtStation:
      state.extinguisherAtStation ?? true,
  };
  const normalizedAgents = state.agents.map((agent, index) => ({
    ...agent,
    orientation: agent.orientation ?? (index < 2 ? "north" : "south"),
    lastAction: agent.lastAction ?? ("STAY" as const),
  }));
  const planningState: MotionState = {
    ...state,
    agents: normalizedAgents,
    counterObjects: worldState.counterObjects,
    cleanPlatesAtRack: worldState.cleanPlatesAtRack,
    grillFood: worldState.grillFood,
    dirtyPlatesInSink: worldState.dirtyPlatesInSink,
    extinguisherAtStation: worldState.extinguisherAtStation,
    dirtyPlatesAtReturn: plateInventory,
  };
  const goals = normalizedAgents.map((agent, index) =>
    index < activeCount
      ? targetForAgent(agent, index, activeCount, map, planningState)
      : null,
  );
  const interactions = normalizedAgents.map((agent, index) =>
    index < activeCount
      ? resolveAgentInteraction(
          agent,
          index,
          activeCount,
          map,
          goals[index],
          plateInventory,
          worldState,
        )
      : {
          agent,
          label: null,
          dishDelta: 0,
          sinkDirtyDelta: 0,
          washedDelta: 0,
          actionKind: null,
        },
  );
  const agents = interactions.map((result) => result.agent);
  const occupied = new Set(
    agents.slice(0, activeCount).map((agent) => pointKey(agent.position)),
  );
  const proposals = agents.map((agent, index) => {
    if (index >= activeCount) return agent.position;
    const occupiedByOthers = new Set(occupied);
    occupiedByOthers.delete(pointKey(agent.position));
    return nextFloorStep(
      agent.position,
      goals[index],
      map,
      blockedCells,
      occupiedByOthers,
    );
  });
  const interactionPause = interactions.map((result) => result.label !== null);
  const canMove = agents.map((agent, index) => {
    if (index >= activeCount) return false;
    const target = proposals[index];
    const distance =
      Math.abs(target[0] - agent.position[0]) +
      Math.abs(target[1] - agent.position[1]);
    return (
      distance === 1 &&
      !blockedCells.has(pointKey(target)) &&
      !interactionPause[index]
    );
  });
  const reasons = state.agents.map(() => null as string | null);
  const blockedKinds = state.agents.map(
    () => null as "collision" | "interaction" | null,
  );
  activeIndices.forEach((index) => {
    if (!canMove[index]) {
      const staying =
        pointKey(proposals[index]) === pointKey(agents[index].position);
      reasons[index] = interactionPause[index]
        ? "执行 PICK/DROP 或 PROCESS"
        : staying
          ? goals[index] === null
            ? "STAY · 当前无可执行任务"
            : "STAY · 等待任务条件"
          : "桌台阻挡";
      blockedKinds[index] =
        interactionPause[index] || staying ? "interaction" : "collision";
    }
  });

  // Resolve only the agents that actually conflict. An agent performing a
  // context action stays in place, but does not globally freeze unrelated
  // teammates elsewhere in the kitchen.
  let conflictChanged = true;
  while (conflictChanged) {
    conflictChanged = false;

    for (let left = 0; left < activeCount; left += 1) {
      for (let right = left + 1; right < activeCount; right += 1) {
        if (!canMove[left] || !canMove[right]) continue;
        const swapsPlaces =
          pointKey(proposals[left]) === pointKey(agents[right].position) &&
          pointKey(proposals[right]) === pointKey(agents[left].position);
        if (!swapsPlaces) continue;

        [left, right].forEach((index) => {
          canMove[index] = false;
          reasons[index] = "迎面换位 · 仅冲突主体等待";
          blockedKinds[index] = "collision";
        });
        conflictChanged = true;
      }
    }

    const targetGroups = new Map<string, number[]>();
    activeIndices.forEach((index) => {
      const target = canMove[index]
        ? proposals[index]
        : agents[index].position;
      const key = pointKey(target);
      targetGroups.set(key, [...(targetGroups.get(key) ?? []), index]);
    });

    targetGroups.forEach((indices) => {
      if (indices.length < 2) return;
      const movingIndices = indices.filter((index) => canMove[index]);
      if (movingIndices.length === 0) return;
      const occupiedByStationaryAgent = indices.some(
        (index) => !canMove[index],
      );

      movingIndices.forEach((index) => {
        canMove[index] = false;
        reasons[index] = occupiedByStationaryAgent
          ? "目标格被占用 · 仅冲突主体等待"
          : "同格冲突 · 仅冲突主体等待";
        blockedKinds[index] = "collision";
      });
      conflictChanged = true;
    });
  }

  const blockedThisStep = activeIndices.filter(
    (index) => blockedKinds[index] === "collision",
  ).length;
  const dirtyPlateDelta = interactions.reduce(
    (total, interaction) => total + interaction.dishDelta,
    0,
  );
  const sinkDirtyPlateDelta = interactions.reduce(
    (total, interaction) => total + interaction.sinkDirtyDelta,
    0,
  );
  const washedPlateDelta = interactions.reduce(
    (total, interaction) => total + interaction.washedDelta,
    0,
  );
  const worldEffects = interactions.flatMap((interaction) => {
    if (!interaction.worldEffect) return [];
    if (
      interaction.actionKind !== "interaction" ||
      !["PICK/DROP", "PROCESS"].includes(interaction.agent.lastAction)
    ) {
      throw new Error("World state mutation requires a valid context action");
    }
    if (
      interaction.worldEffect.grillAction === "take-cooked" &&
      (interaction.agent.carrying === null ||
        !plateBearingItems.has(interaction.agent.carrying))
    ) {
      throw new Error("Cooked beef requires a held plate");
    }
    return [interaction.worldEffect];
  });
  const counterObjects = { ...worldState.counterObjects };
  worldEffects.forEach((effect) => {
    const mutation = effect.counterMutation;
    if (!mutation) return;
    const currentObject = counterObjects[mutation.counterIndex] ?? null;
    if (
      mutation.from === null &&
      mutation.to !== null &&
      currentObject !== null
    ) {
      throw new Error("Counter capacity exceeded: one object per surface cell");
    }
    if (currentObject !== mutation.from) {
      throw new Error("Counter object changed without a matching pickup/drop");
    }
    if (mutation.to === null) {
      delete counterObjects[mutation.counterIndex];
    } else {
      counterObjects[mutation.counterIndex] = mutation.to;
    }
  });
  const cleanPlatesAtRack =
    worldState.cleanPlatesAtRack +
    worldEffects.reduce(
      (total, effect) =>
        total + (effect.cleanPlateDelta ?? 0),
      0,
    );
  if (
    cleanPlatesAtRack < 0 ||
    cleanPlatesAtRack > TOTAL_PHYSICAL_PLATES
  ) {
    throw new Error(
      "Plate rack capacity violated: stack must stay between zero and four",
    );
  }
  const extinguisherStationCount =
    Number(worldState.extinguisherAtStation) +
    worldEffects.reduce(
      (total, effect) =>
        total + (effect.extinguisherStationDelta ?? 0),
      0,
    );
  if (
    extinguisherStationCount < 0 ||
    extinguisherStationCount > 1
  ) {
    throw new Error(
      "Extinguisher station capacity violated: exactly one tool exists",
    );
  }
  const extinguisherAtStation = extinguisherStationCount === 1;
  const grillAction = worldEffects.find((effect) => effect.grillAction)
    ?.grillAction;
  let grillFood = worldState.grillFood;
  let grillCookTicksRemaining = state.grillCookTicksRemaining ?? 0;
  let grillBurnTicksRemaining = state.grillBurnTicksRemaining ?? 0;
  let fireStartedDelta = 0;
  let fireExtinguishedDelta = 0;

  if (grillAction === "place-raw" && grillFood === null) {
    grillFood = "raw-beef";
    grillCookTicksRemaining = GRILL_COOK_STEPS;
    grillBurnTicksRemaining = GRILL_BURN_STEPS;
  } else if (
    grillAction === "take-cooked" &&
    grillFood === "cooked-beef"
  ) {
    grillFood = null;
    grillCookTicksRemaining = 0;
    grillBurnTicksRemaining = 0;
  } else if (
    grillAction === "extinguish" &&
    grillFood === "burnt-beef"
  ) {
    grillFood = null;
    grillCookTicksRemaining = 0;
    grillBurnTicksRemaining = 0;
    fireExtinguishedDelta = 1;
  }

  if (grillFood === "raw-beef") {
    grillCookTicksRemaining = Math.max(0, grillCookTicksRemaining - 1);
    if (grillCookTicksRemaining === 0) {
      grillFood = "cooked-beef";
      grillBurnTicksRemaining = GRILL_BURN_STEPS;
    }
  } else if (grillFood === "cooked-beef") {
    grillBurnTicksRemaining = Math.max(0, grillBurnTicksRemaining - 1);
    if (grillBurnTicksRemaining === 0) {
      grillFood = "burnt-beef";
      fireStartedDelta = 1;
    }
  }

  const nextAgents = agents.map((agent, index) => {
    if (index >= activeCount) return agent;
    const intendedFacing =
      facingToward(agent.position, proposals[index]) ?? agent.orientation;
    if (!canMove[index]) {
      return {
        ...agent,
        orientation: interactionPause[index]
          ? agent.orientation
          : intendedFacing,
        lastAction: interactionPause[index]
          ? agent.lastAction
          : pointKey(proposals[index]) === pointKey(agent.position)
            ? "STAY"
            : facingArrows[intendedFacing],
        blocked: true,
        blockedReason:
          interactions[index].label ?? reasons[index] ?? "保持安全间距",
        blockedKind: blockedKinds[index] ?? "collision",
      };
    }
    return {
      ...agent,
      position: [...proposals[index]] as GridPoint,
      cursor: (agent.cursor + 1) % paths[index].length,
      orientation: intendedFacing,
      lastAction: facingArrows[intendedFacing],
      interactionLabel: null,
      blocked: false,
      blockedReason: null,
      blockedKind: null,
    };
  });
  const nextStep = state.step + 1;
  const scheduledReturnCount = worldEffects.filter(
    (effect) => effect.schedulePlateReturn,
  ).length;
  const pendingPlateReturns = [
    ...(state.pendingPlateReturns ?? []),
    ...Array.from(
      { length: scheduledReturnCount },
      () => state.step + PLATE_RETURN_DELAY_STEPS,
    ),
  ];
  const returnedPlateCount = pendingPlateReturns.filter(
    (dueStep) => dueStep <= nextStep,
  ).length;
  const remainingPlateReturns = pendingPlateReturns.filter(
    (dueStep) => dueStep > nextStep,
  );
  const dirtyPlatesInSink =
    worldState.dirtyPlatesInSink + sinkDirtyPlateDelta;
  if (dirtyPlatesInSink < 0 || dirtyPlatesInSink > 1) {
    throw new Error("Sink capacity violated: exactly one dirty plate at a time");
  }
  const nextState: MotionState = {
    agents: nextAgents,
    step: nextStep,
    avoidedCollisions: state.avoidedCollisions + blockedThisStep,
    dirtyPlatesAtReturn: Math.max(
      0,
      plateInventory + dirtyPlateDelta + returnedPlateCount,
    ),
    dirtyPlatesInSink,
    pendingPlateReturns: remainingPlateReturns,
    cleanPlatesAtRack,
    washedPlates: (state.washedPlates ?? 0) + washedPlateDelta,
    counterObjects,
    discardedItems:
      (state.discardedItems ?? 0) +
      worldEffects.reduce(
        (total, effect) => total + (effect.discardedDelta ?? 0),
        0,
      ),
    firesStarted: (state.firesStarted ?? 0) + fireStartedDelta,
    firesExtinguished:
      (state.firesExtinguished ?? 0) + fireExtinguishedDelta,
    extinguisherAtStation,
    grillFood,
    grillCookTicksRemaining,
    grillBurnTicksRemaining,
    servedOrders:
      (state.servedOrders ?? 0) +
      worldEffects.reduce(
        (total, effect) => total + (effect.servedDelta ?? 0),
        0,
      ),
  };

  if (countPhysicalPlates(nextState) !== TOTAL_PHYSICAL_PLATES) {
    throw new Error("Physical plate conservation invariant violated");
  }
  if (countPhysicalExtinguishers(nextState) !== 1) {
    throw new Error("Physical extinguisher conservation invariant violated");
  }

  return nextState;
}

export default function Home() {
  const [agentCount, setAgentCount] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [configOpen, setConfigOpen] = useState(false);
  const [policyScenario, setPolicyScenario] =
    useState<PolicyScenarioId>("standard");
  const [policyReplays, setPolicyReplays] = useState<
    Partial<Record<PolicyScenarioId, PolicyReplay>>
  >({});
  const [policyFrameIndex, setPolicyFrameIndex] = useState(0);
  const boardRef = useRef<HTMLDivElement>(null);

  const map = maps[0];
  const orthogonalPaths = useMemo(
    () => map.paths.map(orthogonalizePath),
    [map.paths],
  );
  const blockedCells = useMemo(
    () =>
      new Set([
        ...map.counters.map(pointKey),
        ...map.stations.map((station) => pointKey([station.x, station.y])),
      ]),
    [map.counters, map.stations],
  );
  const [motionState, setMotionState] = useState(() =>
    createMotionState(orthogonalPaths),
  );
  const policyReplay = policyReplays[policyScenario] ?? null;
  const policyActive = agentCount === 1 && policyReplay !== null;
  const activePolicyFrame = policyActive
    ? policyReplay.frames[policyFrameIndex] ?? null
    : null;
  const dirtyPlatesAtReturn = Number.isFinite(
    motionState.dirtyPlatesAtReturn,
  )
    ? motionState.dirtyPlatesAtReturn
    : 0;
  const sinkDirtyPlateCount = Number.isFinite(
    motionState.dirtyPlatesInSink,
  )
    ? motionState.dirtyPlatesInSink
    : 0;
  const pendingPlateReturnCount =
    motionState.pendingPlateReturns?.length ?? 0;
  const nextPlateReturnSteps =
    pendingPlateReturnCount > 0
      ? Math.max(
          0,
          Math.min(...motionState.pendingPlateReturns) - motionState.step,
        )
      : 0;
  const cleanPlatesAtRack = Number.isFinite(
    motionState.cleanPlatesAtRack,
  )
    ? motionState.cleanPlatesAtRack
    : 4;
  const extinguisherAtStation =
    motionState.extinguisherAtStation ?? true;
  const physicalPlateCount =
    cleanPlatesAtRack +
    dirtyPlatesAtReturn +
    sinkDirtyPlateCount +
    (motionState.pendingPlateReturns?.length ?? 0) +
    Object.values(motionState.counterObjects ?? {}).reduce(
      (total, item) =>
        total + Number(item !== undefined && plateBearingItems.has(item)),
      0,
    ) +
    motionState.agents.reduce(
      (total, agent) =>
        total +
        Number(
          agent.carrying !== null &&
            plateBearingItems.has(agent.carrying),
        ),
      0,
    );
  const physicalExtinguisherCount =
    countPhysicalExtinguishers(motionState);
  const washedPlates = motionState.washedPlates ?? 0;
  const discardedItems = motionState.discardedItems ?? 0;
  const washingAgent =
    motionState.agents[roleIndices(agentCount).dish];
  const washingActive =
    washingAgent.workKind === "washing" &&
    washingAgent.workTicksRemaining > 0;
  const washingCompletedSteps = washingActive
    ? washingAgent.workTotalTicks - washingAgent.workTicksRemaining
    : 0;
  const washingProgress = washingActive
    ? Math.round(
        (washingCompletedSteps /
          washingAgent.workTotalTicks) *
          100,
      )
    : 0;
  const washingSecondsRemaining = washingActive
    ? washingAgent.workTicksRemaining * 0.42
    : 0;
  const targetOrders = policyActive
    ? policyReplay.deliveries
    : map.results[agentCount - 1];
  const delivered = motionState.servedOrders ?? 0;
  const episodeDurationSeconds = policyActive
    ? (policyReplay.frames.at(-1)?.state.step ?? 0) *
      policyReplay.controlStepSeconds
    : 180;
  const tick = Math.min(
    episodeDurationSeconds,
    motionState.step * CONTROL_STEP_SECONDS,
  );
  const liveRate = tick > 10 ? (delivered / tick) * 60 : 0;
  const progress = tick / Math.max(episodeDurationSeconds, 1);
  const grillFood: FoodKind | null = motionState.grillFood ?? null;
  const grillFailureActive = grillFood === "burnt-beef";
  const grillProgress =
    grillFood === "burnt-beef"
      ? 100
      : grillFood === "raw-beef"
        ? Math.round(
            ((GRILL_COOK_STEPS -
              (motionState.grillCookTicksRemaining ?? GRILL_COOK_STEPS)) /
              GRILL_COOK_STEPS) *
              70,
          )
        : grillFood === "cooked-beef"
          ? Math.round(
              70 +
                ((GRILL_BURN_STEPS -
                  (motionState.grillBurnTicksRemaining ?? GRILL_BURN_STEPS)) /
                  GRILL_BURN_STEPS) *
                  30,
            )
          : 0;
  const grillTicksRemaining =
    grillFood === "raw-beef"
      ? motionState.grillCookTicksRemaining ?? GRILL_COOK_STEPS
      : grillFood === "cooked-beef"
        ? motionState.grillBurnTicksRemaining ?? GRILL_BURN_STEPS
        : 0;
  const grillStageLabel =
    grillFood === "raw-beef"
      ? "煎制中"
      : grillFood === "cooked-beef"
        ? "已熟 · 过火倒计时"
        : grillFood === "burnt-beef"
          ? "烧糊 · 需要灭火"
          : null;
  const counterItems: Partial<
    Record<number, { kind: FoodKind; label: string }>
  > = Object.fromEntries(
    Object.entries(motionState.counterObjects ?? {}).map(([index, kind]) => [
      Number(index),
      {
        kind,
        label: foodLabels[kind],
      },
    ]),
  );
  const agentTask = (index: number, motion: AgentMotion) => {
    if (motion.interactionLabel) return motion.interactionLabel;
    if (policyActive && index === 0) {
      return `PPO POLICY · ${motion.lastAction}`;
    }
    if (motion.carrying) {
      const destinations = [
        "食材桌面",
        "煎台 / 交接桌",
        "餐盘叠放 / 送餐口",
        "洗碗池",
      ];
      return `搬运 ${foodLabels[motion.carrying]} → ${destinations[index]}`;
    }
    if (grillFailureActive && index === 1) return "前往灭火器相邻格";
    return ["前往食材源", "前往牛肉食材源", "前往净盘站", "前往出餐口"][index];
  };
  const agentStatusBadge = (motion: AgentMotion) => {
    if (motion.blockedKind === "collision") return "WAIT";
    if (
      motion.blockedKind === "interaction" &&
      (motion.workKind === "washing" || motion.lastAction === "PROCESS")
    ) {
      return "WORK";
    }

    const label = motion.blockedReason ?? motion.interactionLabel ?? "";
    if (/等待/.test(label)) return "WAIT";
    if (/丢弃|倒掉/.test(label)) return "TRASH";
    if (/取/.test(label)) return "WORK";
    if (/放|归还|送餐/.test(label)) return "DROP";
    return motion.blockedKind === "interaction" ? "ACT" : motion.lastAction;
  };
  const agentStatusText = (motion: AgentMotion) => {
    const badge = agentStatusBadge(motion);
    if (badge === "DROP") return "放置";
    if (badge === "TRASH") return "丢弃";
    if (badge === "WORK") return "持续作业";
    if (badge === "WAIT") return "等待";
    return "操作";
  };
  const scaleRows = map.results.map((orders, index) => {
    const agents = index + 1;
    const throughput = orders / 3;
    const linearBaseline = map.results[0] * agents;
    const synergy =
      index === 0 ? 0 : ((orders - linearBaseline) / linearBaseline) * 100;
    const marginal =
      index === 0
        ? null
        : ((orders - map.results[index - 1]) / map.results[index - 1]) * 100;
    const stage =
      index === 0
        ? "单体基线"
        : index === 1
          ? "形成分工"
          : index === 2
            ? "协同涌现"
            : map.id === "cramped"
              ? "拥堵饱和"
              : "规模扩展";

    return {
      agents,
      orders,
      throughput,
      reward: map.rewards[index],
      synergy,
      marginal,
      stage,
    };
  });
  const selectedScale = scaleRows[agentCount - 1];
  const trainingMaxSteps = ppoTrainingProgress.at(-1)?.steps ?? 1;
  const trainingX = (steps: number) => 74 + (steps / trainingMaxSteps) * 844;
  const trainingY = (deliveriesPerEpisode: number) =>
    246 - (deliveriesPerEpisode / 6) * 202;
  const trainingLinePoints = ppoTrainingProgress
    .map(
      (point) =>
        `${trainingX(point.steps)},${trainingY(point.deliveriesPerEpisode)}`,
    )
    .join(" ");
  const latestTrainingPoint = ppoTrainingProgress.at(-1)!;
  const liveSparseReward =
    activePolicyFrame?.cumulativeSparseReward ?? delivered * 20;
  const liveFirePenalty =
    (motionState.firesStarted ?? 0) * FIRE_STARTED_PENALTY;
  const liveEventReward =
    activePolicyFrame?.cumulativeReward ??
    liveSparseReward + liveFirePenalty;

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      policyScenarioOptions.map(async ({ id, source }) => {
        const response = await fetch(source, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(
            `Policy replay request failed for ${id}: ${response.status}`,
          );
        }
        const replay = (await response.json()) as PolicyReplay;
        if (
          replay.schema !== "nexus.burger.ppo-policy-replay.v1" ||
          replay.scenarioId !== id ||
          replay.frames.length === 0
        ) {
          throw new Error(`Invalid ${id} policy replay`);
        }
        return [id, replay] as const;
      }),
    )
      .then((entries) => {
        if (cancelled) return;
        const replays = Object.fromEntries(entries) as Record<
          PolicyScenarioId,
          PolicyReplay
        >;
        setPolicyReplays(replays);
        setPolicyFrameIndex(0);
        setMotionState(replays.standard.frames[0].state);
      })
      .catch((error: unknown) => {
        console.error("Unable to load PPO policy replays", error);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const replay = policyReplays[policyScenario];
    if (!replay) return;
    setAgentCount(1);
    setPolicyFrameIndex(0);
    setMotionState(replay.frames[0].state);
    setPlaying(true);
  }, [policyReplays, policyScenario]);

  useEffect(() => {
    if (!playing) return;
    if (policyActive) {
      const timer = window.setInterval(() => {
        setPolicyFrameIndex((current) => {
          const next =
            current >= policyReplay.frames.length - 1 ? 0 : current + 1;
          setMotionState(policyReplay.frames[next].state);
          return next;
        });
      }, (policyReplay.controlStepSeconds * 1000) / speed);
      return () => window.clearInterval(timer);
    }
    const timer = window.setInterval(() => {
      setMotionState((current) => {
        const episodeState =
          current.step >= EPISODE_STEPS
            ? createMotionState(orthogonalPaths)
            : current;
        return advanceMotionState(
          episodeState,
          orthogonalPaths,
          agentCount,
          blockedCells,
          map,
        );
      });
    }, (CONTROL_STEP_SECONDS * 1000) / speed);
    return () => window.clearInterval(timer);
  }, [
    agentCount,
    blockedCells,
    map,
    orthogonalPaths,
    playing,
    policyActive,
    policyReplay,
    speed,
  ]);

  const events = useMemo(() => {
    if (policyActive && activePolicyFrame) {
      const eventSummary =
        activePolicyFrame.events.length > 0
          ? activePolicyFrame.events.join(" · ")
          : "no environment event";
      return [
        {
          at: tick,
          tone:
            activePolicyFrame.events.includes("correct_delivery")
              ? "green"
              : "blue",
          title: `PPO ACTION ${activePolicyFrame.action}`,
          meta: `${eventSummary} · step reward ${activePolicyFrame.reward.toFixed(3)}`,
        },
        {
          at: Math.max(0, tick - 2),
          tone: "green",
          title: `${policyReplay.checkpointLabel} · ${policyReplay.validatedSuccesses}/${policyReplay.validatedEpisodes} validated`,
          meta: `${policyReplay.curriculumStage} · deterministic masked argmax · no scripted actions`,
        },
        {
          at: Math.max(0, tick - 4),
          tone: "muted",
          title: `CUMULATIVE REWARD ${activePolicyFrame.cumulativeReward.toFixed(3)}`,
          meta: `sparse ${activePolicyFrame.cumulativeSparseReward.toFixed(1)} · fire ${motionState.firesStarted}`,
        },
      ];
    }
    const base = [
      {
        at: Math.max(2, tick - 2),
        tone: "green",
        title: `MISO-03 完成第 ${Math.max(1, delivered)} 份餐盘叠放`,
        meta: "可见餐盘状态变化 · 仅参与 γΦ(s′)−Φ(s) 状态差分",
      },
      {
        at: Math.max(2, tick - 8),
        tone: "blue",
        title: `MISO-02 → MISO-03 交接熟牛肉`,
        meta: `${map.eventLabel} · 等待 0.8 秒`,
      },
      {
        at: Math.max(2, tick - 15),
        tone: "orange",
        title:
          dirtyPlatesAtReturn > 0
            ? `出餐口已有 ${dirtyPlatesAtReturn} 个脏盘`
            : pendingPlateReturnCount > 0
              ? `${pendingPlateReturnCount} 个餐盘将在 ${(nextPlateReturnSteps * CONTROL_STEP_SECONDS).toFixed(1)} 秒后退出`
              : "出餐口当前为空",
        meta: "送餐口只接收成品 · 脏盘延迟后从独立回收口退出",
      },
      {
        at: Math.max(2, tick - 24),
        tone: "muted",
        title: "局部观测刷新",
        meta: "半径 4 格 · 遮挡开启",
      },
    ];
    if (grillFailureActive) {
      base.unshift({
        at: tick,
        tone: "red",
        title: `煎台过火：团队奖励 ${FIRE_STARTED_PENALTY}`,
        meta: "Failure #F-038 · 灶台已锁定 · MISO-02 正在取灭火器",
      });
    } else if (grillFood === "raw-beef") {
      base.unshift({
        at: tick,
        tone: "orange",
        title: `牛肉煎制中 · 剩余 ${grillTicksRemaining} 步`,
        meta: "环境时钟持续推进 · Agent 可离开煎台执行其他任务",
      });
    } else if (grillFood === "cooked-beef") {
      base.unshift({
        at: tick,
        tone: "green",
        title: `牛肉已熟 · ${grillTicksRemaining} 步后烧糊`,
        meta: "需要及时执行 PICK/DROP 取走熟牛肉",
      });
    }
    if (washingActive) {
      base.unshift({
        at: tick,
        tone: "blue",
        title: `MISO-04 洗盘中 · ${washingProgress}%`,
        meta: `洗碗池占用 · 主体原地等待 · 已洗净 ${washedPlates} 个`,
      });
    }
    if (discardedItems > 0) {
      base.unshift({
        at: tick,
        tone: "muted",
        title: `垃圾桶已处理 ${discardedItems} 份废弃内容`,
        meta: "相邻 PICK/DROP 生效 · 盘中内容清空但实体餐盘保留",
      });
    }
    const interactingAgent = motionState.agents
      .slice(0, agentCount)
      .findIndex((agent) => agent.interactionLabel !== null);
    if (interactingAgent >= 0) {
      const interaction = motionState.agents[interactingAgent];
      base.unshift({
        at: tick,
        tone: "green",
        title: `${teamProfiles[interactingAgent].name} ${interaction.interactionLabel}`,
        meta: `位置 (${interaction.position[0]}, ${interaction.position[1]}) · 四连通相邻校验通过`,
      });
    }
    const waitingAgent = motionState.agents
      .slice(0, agentCount)
      .findIndex((agent) => agent.blockedKind === "collision");
    if (waitingAgent >= 0) {
      base.unshift({
        at: tick,
        tone: "orange",
        title: `${teamProfiles[waitingAgent].name} 检测到占用并原地让行`,
        meta: `${motionState.agents[waitingAgent].blockedReason} · 未进入冲突格`,
      });
    }
    return base;
  }, [
    activePolicyFrame,
    agentCount,
    delivered,
    grillFailureActive,
    grillFood,
    grillTicksRemaining,
    map.eventLabel,
    motionState.agents,
    dirtyPlatesAtReturn,
    discardedItems,
    nextPlateReturnSteps,
    pendingPlateReturnCount,
    policyActive,
    policyReplay,
    tick,
    washedPlates,
    washingActive,
    washingProgress,
  ]);

  function resetEpisode() {
    if (policyActive) {
      setPolicyFrameIndex(0);
      setMotionState(policyReplay.frames[0].state);
      setPlaying(true);
      return;
    }
    setMotionState(createMotionState(orthogonalPaths));
    setPlaying(true);
  }

  function injectFault() {
    setMotionState((current) => {
      if (current.grillFood === "burnt-beef") return current;
      return {
        ...current,
        grillFood: "burnt-beef",
        grillCookTicksRemaining: 0,
        grillBurnTicksRemaining: 0,
        firesStarted: (current.firesStarted ?? 0) + 1,
      };
    });
    setPlaying(true);
  }

  function exportReplay() {
    if (policyActive) {
      const blob = new Blob([JSON.stringify(policyReplay, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `ppo-${policyReplay.checkpointLabel}-${policyReplay.curriculumStage}.json`;
      anchor.click();
      URL.revokeObjectURL(url);
      return;
    }
    const payload = {
      schema: "nexus.burger.replay.v1",
      generated_at: new Date().toISOString(),
      policy:
        "Deterministic role policy / standalone official-gameplay-derived core",
      map: map.id,
      agents: agentCount,
      episode_seconds: tick,
      delivered,
      failures_recovered:
        BASELINE_RECOVERED_FAULTS +
        (motionState.firesExtinguished ?? 0),
      collisions_avoided: motionState.avoidedCollisions,
      discarded_items: discardedItems,
      dirty_plates_at_return: dirtyPlatesAtReturn,
      dirty_plates_in_sink: sinkDirtyPlateCount,
      pending_plate_returns: motionState.pendingPlateReturns,
      plate_return_delay_steps: PLATE_RETURN_DELAY_STEPS,
      local_observation_radius: 4,
      action_space: [
        "north",
        "south",
        "east",
        "west",
        "stay",
        "pick_drop",
        "process",
      ],
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `nexus-${map.id}-${agentCount}agents-replay.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function fullscreenBoard() {
    if (boardRef.current?.requestFullscreen) {
      await boardRef.current.requestFullscreen();
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <button
          className="brand"
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
          aria-label="返回顶部"
        >
          <span className="brand-mark">
            <i />
            <i />
            <i />
          </span>
          <span>
            <strong>NEXUS</strong>
            <small>MULTI-AGENT LAB</small>
          </span>
        </button>

        <nav className="main-nav" aria-label="主导航">
          <button className="active">仿真运行</button>
          <button
            onClick={() =>
              document
                .getElementById("benchmark")
                ?.scrollIntoView({ behavior: "smooth" })
            }
          >
            评测对比
          </button>
          <button
            onClick={() =>
              document
                .getElementById("closed-loop")
                ?.scrollIntoView({ behavior: "smooth" })
            }
          >
            失败回流
          </button>
        </nav>

        <div className="top-actions">
          <span className="demo-badge">
            <i />
            {policyActive
              ? `PPO POLICY · ${policyReplay.scenarioLabel} · ${policyReplay.validatedSuccesses}/${policyReplay.validatedEpisodes} VALIDATED`
              : "DETERMINISTIC ROLE POLICY · NO PPO/MAPPO"}
          </span>
          <button className="ghost-button" onClick={exportReplay}>
            <span aria-hidden="true">↓</span> 导出回放
          </button>
          <button className="primary-button" onClick={() => setConfigOpen(true)}>
            MAPPO 配置
          </button>
        </div>
      </header>

      <section className="run-header">
        <div>
          <span className="eyebrow">EXPERIMENT / OC-BURGER-003</span>
          <h1>汉堡协作厨房</h1>
          <p>
            局部感知 · 去中心化执行 · 集中式训练
            <span className="separator">/</span>
            STANDALONE BURGER CORE · 7-ACTION PPO/MAPPO INTERFACE
          </p>
        </div>
        <div className="run-status">
          <span className="live-dot" />
          <div>
            <small>EPISODE STATUS</small>
            <strong>
              {playing
                ? policyActive
                  ? "PPO 策略回放"
                  : "确定性演示回放"
                : "已暂停"}
            </strong>
          </div>
          <span className="run-id">RUN 0240</span>
        </div>
      </section>

      <section className="workspace">
        <aside className="left-rail panel">
          <div className="panel-title">
            <div>
              <span className="index">01</span>
              <h2>环境配置</h2>
            </div>
            <span className="muted-label">1 SCENE</span>
          </div>

          <div className="control-block">
            <label>当前仿真地图</label>
            <div className="scenario-focus-card">
              <span className="mini-map mini-map-cramped">
                <i />
                <i />
                <i />
              </span>
              <div>
                <strong>{map.shortName}</strong>
                <small>{map.title}</small>
              </div>
              <b>01</b>
            </div>
            <p className="scenario-note">
              参考 Overcooked 官方紧凑关卡：外围连续工作台包围中央实体岛台，只保留左右两个单格交汇口。
            </p>
          </div>

          <div className="control-block replay-scenario-block">
            <label>PPO 开局场景</label>
            <div
              className="policy-scenario-options"
              role="tablist"
              aria-label="选择 PPO 开局场景"
            >
              {policyScenarioOptions.map((scenario) => {
                const replay = policyReplays[scenario.id];
                const selected = policyScenario === scenario.id;
                return (
                  <button
                    key={scenario.id}
                    type="button"
                    role="tab"
                    aria-selected={selected}
                    className={selected ? "selected" : ""}
                    onClick={() => setPolicyScenario(scenario.id)}
                  >
                    <span
                      className={`scenario-glyph scenario-glyph-${scenario.id}`}
                      aria-hidden="true"
                    >
                      {scenario.id === "standard"
                        ? "●"
                        : scenario.id === "fire"
                          ? "▲"
                          : "≋"}
                    </span>
                    <span>
                      <strong>{scenario.label}</strong>
                      <small>{scenario.description}</small>
                    </span>
                    <b>{scenario.index}</b>
                    <i className="scenario-load-state">
                      {replay ? "READY" : "LOAD"}
                    </i>
                  </button>
                );
              })}
            </div>
            <div className="policy-scenario-summary">
              <span>
                {policyReplay
                  ? `${policyReplay.deliveries} 份送餐`
                  : "正在载入 PPO 回放"}
              </span>
              {policyReplay && (
                <span>
                  {policyScenario === "fire"
                    ? `灭火 ${policyReplay.firesExtinguished} 次`
                    : `洗净 ${policyReplay.washedPlates} 个盘子`}
                </span>
              )}
              <b>POLICY ONLY</b>
            </div>
          </div>

          <div className="control-block workflow-block">
            <label>汉堡任务流程</label>
            <div className="workflow-step">
              <span>1</span>
              <div>
                <strong>桌面摆料</strong>
                <small>
                  散装食材占用一个桌格 · 食材源可直接加入手持餐盘
                </small>
              </div>
            </div>
            <div className="workflow-step">
              <span>2</span>
              <div>
                <strong>煎制</strong>
                <small>
                  煎制 24 步（10.1 秒）· 熟后 16 步（6.7 秒）内必须取走
                </small>
              </div>
            </div>
            <div className="workflow-step">
              <span>3</span>
              <div>
                <strong>餐盘逐件叠放</strong>
                <small>
                  手持半成品可在食材源或煎台直接合并 · 全程显示餐盘
                </small>
              </div>
            </div>
            <div className="workflow-step">
              <span>4</span>
              <div>
                <strong>送餐回收</strong>
                <small>送餐口收走成品 → 5.0 秒后脏盘从独立出餐口退出</small>
              </div>
            </div>
            <div className="workflow-step">
              <span>5</span>
              <div>
                <strong>废弃处理</strong>
                <small>散装食材可丢弃 · 盘中内容倒空后保留实体净盘</small>
              </div>
            </div>
          </div>

          <div className="control-block">
            <label>主体数量</label>
            <div className="agent-segment" aria-label="选择主体数量">
              {[1, 2, 3, 4].map((count) => (
                <button
                  key={count}
                  className={agentCount === count ? "selected" : ""}
                  onClick={() => {
                    setAgentCount(count);
                    if (count === 1 && policyReplay) {
                      setPolicyFrameIndex(0);
                      setMotionState(policyReplay.frames[0].state);
                    } else {
                      setMotionState(createMotionState(orthogonalPaths));
                    }
                  }}
                >
                  {count}
                </button>
              ))}
            </div>
            <div className="segment-caption">
              <span>单主体基线</span>
              <span>协作饱和点</span>
            </div>
          </div>

          <div className="control-block observation-block">
            <label>局部感知</label>
            <div className="setting-row">
              <span>
                <i className="setting-icon">⌗</i>
                观测半径
              </span>
              <strong>4 格</strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">◐</i>
                遮挡模拟
              </span>
              <span className="toggle on">
                <i />
              </span>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">■</i>
                桌台 / 灶台碰撞
              </span>
              <strong>硬约束</strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">●</i>
                同格终点 / 迎面换位
              </span>
              <strong>仅冲突主体等待</strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">◎</i>
                交互判定
              </span>
              <strong>相邻即可 · 无需朝向</strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">○</i>
                熟牛肉取用
              </span>
              <strong>必须用餐盘承接</strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">≈</i>
                清洗容量
              </span>
              <strong>1 盘 / 4.2 秒</strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">↩</i>
                退盘延迟
              </span>
              <strong>12 步 / 5.0 秒</strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">✓</i>
                奖励结算
              </span>
              <strong>仅环境事件</strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">#</i>
                状态完整性
              </span>
              <strong>每步校验 + 摘要</strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">○</i>
                实体盘子守恒
              </span>
              <strong>
                {physicalPlateCount} / {TOTAL_PHYSICAL_PLATES}
              </strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">灭</i>
                实体灭火器守恒
              </span>
              <strong>{physicalExtinguisherCount} / 1</strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">≡</i>
                净盘叠放
              </span>
              <strong>盘架 0–4 · 食品单格</strong>
            </div>
            <div className="setting-row">
              <span>
                <i className="setting-icon">⌁</i>
                通信信道
              </span>
              <strong>无</strong>
            </div>
          </div>

          <div className="objective-card">
            <span>TEAM OBJECTIVE</span>
            <strong>180 秒内最大化正确出餐</strong>
            <p>共享团队奖励；只奖励合法出餐，碰撞、时间与首次起火受罚。</p>
          </div>
        </aside>

        <section className="simulation-column">
          <div className="simulation-card panel">
            <div className="simulation-head">
              <div>
                <span className="map-number">MAP 01 · SINGLE SCENE</span>
                <h2>{map.title}</h2>
                <p>{map.subtitle}</p>
              </div>
              <div className="sim-head-actions">
                {policyActive && (
                  <span className="policy-chip">
                    PPO · {policyReplay.curriculumStage.toUpperCase()}
                  </span>
                )}
                <span className="view-mode-chip">
                  <i aria-hidden="true" />
                  2.5D 立体视角
                </span>
                <span className="scenario-chip">
                  <i style={{ background: map.accent }} />
                  {map.bottleneck}
                </span>
                <button onClick={fullscreenBoard} aria-label="全屏查看仿真">
                  ⛶
                </button>
              </div>
            </div>

            <div
              className={`kitchen-board kitchen-${map.id}`}
              ref={boardRef}
              style={
                {
                  "--map-accent": map.accent,
                  "--cell-w": `${100 / map.cols}%`,
                  "--cell-h": `${100 / map.rows}%`,
                } as CSSProperties
              }
            >
              <div className="floor-grid" />

              {map.counters.map(([x, y], index) => (
                <div
                  className="counter"
                  key={`counter-${index}`}
                  data-obstacle="counter"
                  style={{
                    left: `${(x / map.cols) * 100}%`,
                    top: `${(y / map.rows) * 100}%`,
                    zIndex: 2 + y * 2,
                  }}
                >
                  <span className="counter-front" aria-hidden="true" />
                  {counterItems[index] && (
                    <span
                      className="counter-surface-item"
                      data-counter-object={counterItems[index].kind}
                      title={counterItems[index].label}
                      style={
                        {
                          "--counter-item-x": "50%",
                          "--counter-item-scale": "0.92",
                        } as CSSProperties
                      }
                    >
                      <FoodArt kind={counterItems[index].kind} />
                    </span>
                  )}
                </div>
              ))}

              {map.stations.map((station, index) => {
                const meta = stationMeta[station.kind];
                return (
                  <div
                    className={`station station-${station.kind} ${
                      station.kind === "grill" && grillFailureActive
                        ? "grill-disabled"
                        : ""
                    } ${
                      station.kind === "extinguisher" &&
                      !extinguisherAtStation
                        ? "extinguisher-empty"
                        : ""
                    }`}
                    key={`${station.kind}-${index}`}
                    data-obstacle={station.kind}
                    data-grill-usable={
                      station.kind === "grill"
                        ? String(!grillFailureActive)
                        : undefined
                    }
                    data-extinguisher-available={
                      station.kind === "extinguisher"
                        ? String(extinguisherAtStation)
                        : undefined
                    }
                    style={{
                      left: `${(station.x / map.cols) * 100}%`,
                      top: `${(station.y / map.rows) * 100}%`,
                      zIndex: 2 + station.y * 2,
                    }}
                    title={
                      station.kind === "extinguisher"
                        ? extinguisherAtStation
                          ? "唯一灭火器 · 在位"
                          : "唯一灭火器 · 已被主体取用"
                        : meta.label
                    }
                  >
                    <span className="station-front" aria-hidden="true" />
                    <span
                      className={`station-pictogram pictogram-${station.kind}`}
                      aria-hidden="true"
                    >
                      <i />
                      <i />
                      <i />
                    </span>
                    {station.kind === "plate" &&
                      cleanPlatesAtRack > 0 && (
                        <span
                          className="returned-dishes rack-dishes"
                          title={`${cleanPlatesAtRack} 个可用净盘`}
                        >
                          <span
                            className="dish-stack"
                            aria-hidden="true"
                          >
                            {Array.from(
                              {
                                length: Math.min(
                                  cleanPlatesAtRack,
                                  4,
                                ),
                              },
                              (_, layer) => (
                                <span
                                  className="return-dish-layer"
                                  key={layer}
                                  style={{
                                    left: `${layer * 2}px`,
                                    top: `${-layer * 4}px`,
                                    zIndex: layer + 1,
                                  }}
                                >
                                  <FoodArt kind="clean-plate" />
                                </span>
                              ),
                            )}
                          </span>
                          <b>×{cleanPlatesAtRack}</b>
                        </span>
                      )}
                    {station.kind === "grill" && (
                      <>
                        {grillFood && (
                          <>
                            <span className="grill-food">
                              <FoodArt kind={grillFood} />
                            </span>
                            <span
                              className={`grill-steam ${grillFailureActive ? "danger" : ""}`}
                              aria-hidden="true"
                            >
                              <i />
                              <i />
                              <i />
                            </span>
                          </>
                        )}
                        <i
                          className={`grill-progress ${grillFailureActive ? "danger" : ""}`}
                          style={{ "--progress": `${grillProgress}%` } as CSSProperties}
                        />
                        {grillStageLabel && (
                          <span
                            className={`grill-status ${grillFailureActive ? "danger" : ""}`}
                            aria-label={grillStageLabel}
                          >
                            <b>{grillStageLabel}</b>
                            <small>
                              {grillFailureActive
                                ? `灶台锁定 · 等待灭火器 · ${FIRE_STARTED_PENALTY}`
                                : `${grillTicksRemaining} 步 · ${(grillTicksRemaining * CONTROL_STEP_SECONDS).toFixed(1)}s`}
                            </small>
                          </span>
                        )}
                      </>
                    )}
                    {station.kind === "return" &&
                      dirtyPlatesAtReturn > 0 && (
                        <span
                          className="returned-dishes"
                          title={`${dirtyPlatesAtReturn} 个待清洗脏盘`}
                        >
                          <span className="dish-stack" aria-hidden="true">
                            {Array.from(
                              {
                                length: Math.min(
                                  dirtyPlatesAtReturn,
                                  4,
                                ),
                              },
                              (_, layer) => (
                                <span
                                  className="return-dish-layer"
                                  key={layer}
                                  style={{
                                    left: `${layer * 2}px`,
                                    top: `${-layer * 4}px`,
                                    zIndex: layer + 1,
                                  }}
                                >
                                  <FoodArt kind="dirty-plate" />
                                </span>
                              ),
                            )}
                          </span>
                          <b>×{dirtyPlatesAtReturn}</b>
                        </span>
                      )}
                    {station.kind === "return" &&
                      dirtyPlatesAtReturn === 0 &&
                      pendingPlateReturnCount > 0 && (
                        <span className="return-pending">
                          ↻{" "}
                          {(
                            nextPlateReturnSteps *
                            CONTROL_STEP_SECONDS
                          ).toFixed(1)}
                          s
                        </span>
                      )}
                    {station.kind === "sink" &&
                      sinkDirtyPlateCount > 0 && (
                        <span
                          className="sink-dish"
                          data-dirty-count={sinkDirtyPlateCount}
                          title={`水槽内有 ${sinkDirtyPlateCount} 个脏盘`}
                        >
                          <FoodArt kind="dirty-plate" />
                        </span>
                      )}
                    {station.kind === "sink" && washingActive && (
                      <>
                        <span className="wash-bubbles" aria-hidden="true">
                          <i />
                          <i />
                          <i />
                        </span>
                        <span
                          className="wash-status"
                          aria-label={`洗盘进度 ${washingProgress}% · 剩余 ${washingSecondsRemaining.toFixed(1)} 秒`}
                        >
                          <span className="wash-status-heading">
                            <small>WASHING</small>
                            <b>{washingProgress}%</b>
                          </span>
                          <i
                            className="wash-progress"
                            role="progressbar"
                            aria-label="Dish washing progress"
                            aria-valuemin={0}
                            aria-valuemax={WASH_DURATION_STEPS}
                            aria-valuenow={washingCompletedSteps}
                          >
                            <em style={{ width: `${washingProgress}%` }} />
                          </i>
                          <span className="wash-status-meta">
                            <small>
                              STEP {washingCompletedSteps} /{" "}
                              {WASH_DURATION_STEPS}
                            </small>
                            <b>{washingSecondsRemaining.toFixed(1)}s</b>
                          </span>
                        </span>
                      </>
                    )}
                    {station.kind === "sink" &&
                      sinkDirtyPlateCount > 0 && (
                      <span
                        className="sink-occupancy"
                        aria-label={`洗碗池内有 ${sinkDirtyPlateCount} 个脏盘，一次清洗 1 个`}
                      >
                        <small>DIRTY</small>
                        <b>×{sinkDirtyPlateCount}</b>
                      </span>
                      )}
                  </div>
                );
              })}

              {grillFailureActive && (
                <div className="fire-alert" role="status" aria-live="polite">
                  <span className="flame" aria-hidden="true">🔥</span>
                  <strong>过火</strong>
                  <small>灶台停用 · 团队奖励 {FIRE_STARTED_PENALTY}</small>
                </div>
              )}

              {teamProfiles.slice(0, agentCount).map((agent, index) => {
                const motion = motionState.agents[index];
                const point = motion.position;
                const carry = motion.carrying;
                const orientation =
                  motion.orientation ?? (index < 2 ? "north" : "south");
                const lastAction = motion.lastAction ?? "STAY";
                return (
                  <div
                    className={`agent facing-${orientation} ${
                      motion.blockedKind === "collision"
                        ? "collision-wait"
                        : motion.blockedKind === "interaction"
                          ? "interaction-wait"
                          : ""
                    }`}
                    key={agent.name}
                    data-action={lastAction}
                    data-orientation={orientation}
                    style={
                      {
                        "--agent-color": agent.color,
                        left: `${((point[0] + 0.5) / map.cols) * 100}%`,
                        top: `${((point[1] + 0.5) / map.rows) * 100}%`,
                        zIndex: 3 + point[1] * 2,
                      } as CSSProperties
                    }
                  >
                    {carry && (
                      <span
                        className={`carried-item ${carry === "extinguisher" ? "carried-tool" : ""}`}
                        title={`手持：${foodLabels[carry]}`}
                      >
                        <span className="carry-hands" />
                        <FoodArt kind={carry} />
                      </span>
                    )}
                    <span className="agent-shadow" />
                    <span className="facing-indicator" aria-hidden="true">
                      {facingArrows[orientation]}
                    </span>
                    <span className="agent-action-chip">{lastAction}</span>
                    {motion.blocked && (
                      <span
                        className="collision-ring"
                        aria-label={motion.blockedReason ?? "等待"}
                      >
                        {agentStatusBadge(motion)}
                      </span>
                    )}
                    <span className="chef-character">
                      <i className="chef-hat" />
                      <i className="chef-face">
                        <em />
                        <em />
                      </i>
                      <i className="chef-body" />
                    </span>
                    <small>{index + 1}</small>
                    <b>
                      {motion.blocked
                        ? motion.blockedKind === "interaction"
                          ? `${agentStatusText(motion)} · ${motion.blockedReason}`
                          : `让行 · ${motion.blockedReason}`
                        : agentTask(index, motion)}
                    </b>
                    {index === 2 && tick % 38 < 8 && agentCount > 2 && (
                      <span className="action-pop">交接完成</span>
                    )}
                  </div>
                );
              })}

            </div>

            <div className="below-board-hud">
              <div className="game-score">
                <span>总积分 · TEAM SCORE</span>
                <strong>{formatScore(liveEventReward)}</strong>
                <small>★ 连击 × {Math.max(1, Math.min(4, delivered))}</small>
              </div>

              <div className="order-queue" aria-label="当前订单队列">
                {[0, 1, 2].map((order) => (
                  <div
                    className={`order-ticket ${order === 0 ? "current" : ""}`}
                    key={order}
                  >
                    <span>{order === 0 ? "NOW" : `0${order + 1}`}</span>
                    <div className="ticket-burger">
                      <i />
                      <i />
                      <i />
                      <i />
                    </div>
                    <div className="ticket-copy">
                      <strong>经典汉堡</strong>
                      <small>
                        <i className="ingredient bun-dot" />
                        <i className="ingredient lettuce-dot" />
                        <i className="ingredient beef-dot" />
                        <i className="ingredient plate-dot" />
                      </small>
                    </div>
                    <b
                      style={{
                        width: `${Math.max(16, 76 - order * 23 - (tick % 18))}%`,
                      }}
                    />
                  </div>
                ))}
              </div>

              <div className="game-timer">
                <span>TIME LEFT</span>
                <strong>{formatTime(episodeDurationSeconds - tick)}</strong>
                <i style={{ width: `${(1 - progress) * 100}%` }} />
              </div>

              <span className="interaction-strip">
                ↑ ↓ ← → · STAY · PICK/DROP · PROCESS · 相邻即可，无需转身
              </span>
            </div>

            <div className="playback-bar">
              <button className="round-control" onClick={resetEpisode} title="重置">
                ↺
              </button>
              <button
                className="play-control"
                onClick={() => setPlaying((value) => !value)}
                aria-label={playing ? "暂停" : "播放"}
              >
                {playing ? "Ⅱ" : "▶"}
              </button>
              <span className="timecode">{formatTime(tick)}</span>
              <div className="timeline">
                <i style={{ width: `${progress * 100}%` }} />
                <b style={{ left: `${progress * 100}%` }} />
                <span className="failure-marker" style={{ left: "68%" }} />
              </div>
              <span className="duration">
                {formatTime(episodeDurationSeconds)}
              </span>
              <button
                className="speed-button"
                onClick={() => setSpeed((value) => (value === 4 ? 1 : value * 2))}
              >
                {speed}×
              </button>
            </div>

            <div className="live-kpis">
              <div>
                <span>已完成订单</span>
                <strong>{delivered}</strong>
                <small>
                  / {targetOrders} {policyActive ? "验证轨迹" : "目标轨迹"}
                </small>
              </div>
              <div>
                <span>实时吞吐</span>
                <strong>{liveRate.toFixed(2)}</strong>
                <small>份 / 分钟</small>
              </div>
              <div>
                <span>累计事件奖励</span>
                <strong>{liveEventReward.toFixed(2)}</strong>
                <small>
                  出餐 {liveSparseReward.toFixed(1)} · 起火 {liveFirePenalty}
                </small>
              </div>
              <div>
                <span>协同指数</span>
                <strong>{map.coordination[agentCount - 1]}</strong>
                <small>/ 100</small>
              </div>
              <div>
                <span>碰撞避免</span>
                <strong>{motionState.avoidedCollisions}</strong>
                <small>次 · 同格与穿越均阻止</small>
              </div>
              <div>
                <span>废弃处理</span>
                <strong>{discardedItems}</strong>
                <small>份 · 无直接正奖励</small>
              </div>
            </div>
          </div>
        </section>

        <aside className="right-rail">
          <section className="panel agents-panel">
            <div className="panel-title">
              <div>
                <span className="index">02</span>
                <h2>主体状态</h2>
              </div>
              <span className="healthy">
                <i />
                {agentCount}/{agentCount} HEALTHY
              </span>
            </div>
            <div className="agent-list">
              {teamProfiles.slice(0, agentCount).map((agent, index) => {
                const motion = motionState.agents[index];
                const carry = motion.carrying;
                return (
                  <div className="agent-row" key={agent.name}>
                    <span
                      className="agent-avatar"
                      style={{ "--agent-color": agent.color } as CSSProperties}
                    >
                      {index + 1}
                    </span>
                    <div>
                      <strong>{agent.name}</strong>
                      <small>{agent.role}</small>
                    </div>
                    <span className="agent-task">
                      <b>
                        {motion.blocked
                          ? motion.blockedKind === "interaction"
                            ? `${agentStatusText(motion)} · ${motion.blockedReason}`
                            : `让行 · ${motion.blockedReason}`
                          : agentTask(index, motion)}
                      </b>
                      <small>
                        {carry ? `手持 ${foodLabels[carry]}` : "双手空闲"} ·{" "}
                        {facingArrows[motion.orientation ?? (index < 2 ? "north" : "south")]}{" "}
                        {motion.lastAction ?? "STAY"}
                      </small>
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="role-balance">
              <div>
                <span>角色专精度</span>
                <strong>{benchmarks[agentCount - 1].specialization}%</strong>
              </div>
              <div className="balance-bar">
                {teamProfiles.slice(0, agentCount).map((agent) => (
                  <i
                    key={agent.name}
                    style={{
                      background: agent.color,
                      width: `${100 / agentCount}%`,
                    }}
                  />
                ))}
              </div>
              <small>基于 action–station 互信息的演示估计</small>
            </div>
          </section>

          <section className="panel event-panel">
            <div className="panel-title">
              <div>
                <span className="index">03</span>
                <h2>协作事件流</h2>
              </div>
              <span className="muted-label">LIVE</span>
            </div>
            <div className="event-list">
              {events.map((event, index) => (
                <div className="event-row" key={`${event.title}-${index}`}>
                  <span className={`event-dot ${event.tone}`} />
                  <div>
                    <strong>{event.title}</strong>
                    <small>{event.meta}</small>
                  </div>
                  <time>{formatTime(event.at)}</time>
                </div>
              ))}
            </div>
            <button
              className={`fault-button ${grillFailureActive ? "active" : ""}`}
              onClick={injectFault}
              disabled={grillFailureActive || policyActive}
            >
              <span aria-hidden="true">🔥</span>
              <div>
                <strong>
                  {policyActive
                    ? "PPO 回放保持只读"
                    : grillFailureActive
                      ? "恢复策略执行中"
                      : "注入煎台过火故障"}
                </strong>
                <small>
                  {policyActive
                    ? "切换到 2–4 主体演示后可注入"
                    : "生成可回流 Failure Case"}
                </small>
              </div>
            </button>
          </section>
        </aside>
      </section>

      <section className="benchmark-section" id="benchmark">
        <div className="section-heading">
          <div>
            <span className="eyebrow">SCALING STUDY / 01–04 AGENTS</span>
            <h2>协作规模与边际效应</h2>
            <p>
              演示基线用于定义评测协议，不代表真实 MAPPO 训练结果。正式实验需报告
              5 个随机种子的均值与 95% 置信区间。
            </p>
          </div>
          <div className="legend">
            <span>
              <i className="legend-bar" /> 送餐 / 分钟
            </span>
            <span>
              <i className="legend-reward" /> 团队总奖励 / 180 秒
            </span>
            <span>
              <i className="legend-line" /> 协同增益
            </span>
          </div>
        </div>

        <div className="benchmark-grid">
          <div className="training-progress-card panel">
            <div className="training-progress-head">
              <div>
                <span>SINGLE-AGENT PPO · EVALUATED CHECKPOINTS</span>
                <strong>训练步数与每局平均送餐数</strong>
                <small>
                  主指标为固定 429-step（约 180 秒）评估中的平均送餐数；每分钟送餐数
                  仅作时长归一化参考。虚线前为固定开局，之后均为随机位置评估。
                </small>
              </div>
              <div className="training-latest">
                <span>最新策略</span>
                <strong>
                  {latestTrainingPoint.deliveriesPerEpisode.toFixed(2)}
                </strong>
                <small>
                  份 / 局 · {latestTrainingPoint.dpm.toFixed(3)} 份 / 分钟
                </small>
                <small>标准 5.675 · 脏盘 5.225 · 缺盘 4.700 · 过火 5.250</small>
              </div>
            </div>

            <div className="training-chart-scroll">
              <svg
                className="training-progress-chart"
                viewBox="0 0 960 304"
                role="img"
                aria-labelledby="ppo-progress-title ppo-progress-description"
              >
                <title id="ppo-progress-title">
                  单智能体 PPO 每局平均送餐数随训练步数变化
                </title>
                <desc id="ppo-progress-description">
                  五个连续训练阶段共采样 1179648 个环境步。最新 checkpoint 在标准、
                  脏盘、缺盘与过火各 40 个固定位置上的联合均值为每局 5.2125 份。
                </desc>

                {trainingPhases.map((phase, index) => {
                  const x = trainingX(phase.start);
                  const width = trainingX(phase.end) - x;
                  return (
                    <g key={phase.label}>
                      <rect
                        className={`training-phase-band phase-band-${index + 1}`}
                        x={x}
                        y="20"
                        width={width}
                        height="226"
                      />
                      <text
                        className="training-phase-label"
                        x={x + width / 2}
                        y="34"
                        textAnchor="middle"
                      >
                        {phase.label}
                      </text>
                    </g>
                  );
                })}

                {[0, 1.5, 3, 4.5, 6].map((tickValue) => (
                  <g key={`y-${tickValue}`}>
                    <line
                      className="training-gridline"
                      x1="74"
                      x2="918"
                      y1={trainingY(tickValue)}
                      y2={trainingY(tickValue)}
                    />
                    <text
                      className="training-axis-label"
                      x="62"
                      y={trainingY(tickValue) + 3}
                      textAnchor="end"
                    >
                      {tickValue.toFixed(2)}
                    </text>
                  </g>
                ))}

                {[0, 262_144, 524_288, 786_432, 1_048_576, 1_179_648].map(
                  (tickValue) => (
                    <g key={`x-${tickValue}`}>
                      <line
                        className="training-axis-tick"
                        x1={trainingX(tickValue)}
                        x2={trainingX(tickValue)}
                        y1="246"
                        y2="252"
                      />
                      <text
                        className="training-axis-label"
                        x={trainingX(tickValue)}
                        y="268"
                        textAnchor="middle"
                      >
                        {tickValue === 0
                          ? "0"
                          : tickValue >= 1_000_000
                            ? `${(tickValue / 1_000_000).toFixed(2)}M`
                            : `${Math.round(tickValue / 1_000)}k`}
                      </text>
                    </g>
                  ),
                )}

                <line
                  className="training-protocol-boundary"
                  x1={trainingX(262_144)}
                  x2={trainingX(262_144)}
                  y1="20"
                  y2="246"
                />
                <text
                  className="training-boundary-label"
                  x={trainingX(262_144) + 7}
                  y="57"
                >
                  切换为随机位置评估
                </text>

                <polyline
                  className="training-progress-line"
                  points={trainingLinePoints}
                />
                {ppoTrainingProgress.map((point, index) => (
                  <g key={`${point.steps}-${point.phase}`}>
                    <circle
                      className={
                        index === ppoTrainingProgress.length - 1
                          ? "training-progress-point latest"
                          : "training-progress-point"
                      }
                      cx={trainingX(point.steps)}
                      cy={trainingY(point.deliveriesPerEpisode)}
                      r={index === ppoTrainingProgress.length - 1 ? 5 : 3.5}
                    >
                      <title>
                        {`${point.phase} · ${point.steps.toLocaleString()} 步 · ${point.deliveriesPerEpisode.toFixed(2)} 份/局 · ${point.dpm.toFixed(3)} 份/分钟 · ${point.protocol}`}
                      </title>
                    </circle>
                  </g>
                ))}

                <line
                  className="training-latest-guide"
                  x1={trainingX(latestTrainingPoint.steps)}
                  x2={trainingX(latestTrainingPoint.steps)}
                  y1={trainingY(latestTrainingPoint.deliveriesPerEpisode)}
                  y2="246"
                />
                <text
                  className="training-latest-label"
                  x={trainingX(latestTrainingPoint.steps) - 8}
                  y={trainingY(latestTrainingPoint.deliveriesPerEpisode) - 11}
                  textAnchor="end"
                >
                  最新 {latestTrainingPoint.deliveriesPerEpisode.toFixed(2)} 份/局
                </text>
                <text
                  className="training-axis-title"
                  x="496"
                  y="294"
                  textAnchor="middle"
                >
                  累计采样环境步数
                </text>
              </svg>
            </div>
            <div className="training-progress-foot">
              <span>
                <i className="progress-line-key" /> 每局平均送餐数（主指标）
              </span>
              <span>
                <i className="protocol-key" /> 评估协议切换
              </span>
              <p>
                786k 处的回落触发回滚到该阶段最佳 checkpoint，再进行困难开局强化；
                曲线保留该回落，避免只展示最佳结果。
              </p>
            </div>
          </div>

          <div className="scale-card panel">
            <div className="scale-card-head">
              <div>
                <span>TEAM SCALING CURVE · {map.title.toUpperCase()}</span>
                <strong>主体数量增加，是否真正产生“1 + 1 &gt; 2”的协同效应？</strong>
              </div>
              <div className="scale-selected-kpis">
                <div>
                  <span>当前切片</span>
                  <strong>{agentCount} 主体</strong>
                </div>
                <div>
                  <span>单位时间送餐</span>
                  <strong>{selectedScale.throughput.toFixed(2)}</strong>
                  <small>份 / 分钟</small>
                </div>
                <div>
                  <span>团队总奖励</span>
                  <strong>{selectedScale.reward}</strong>
                  <small>/ 180 秒</small>
                </div>
                <div
                  className={
                    selectedScale.synergy < 0 ? "negative-effect" : "positive-effect"
                  }
                >
                  <span>协同增益</span>
                  <strong>
                    {agentCount === 1
                      ? "基线"
                      : signedPercent(selectedScale.synergy)}
                  </strong>
                  <small>相对 N × 单主体</small>
                </div>
              </div>
            </div>

            <div className="dual-chart">
              <div className="scale-axis left-axis">
                <strong>送餐 / 分钟</strong>
                <span>5.0</span>
                <span>3.75</span>
                <span>2.5</span>
                <span>1.25</span>
                <span>0</span>
              </div>
              <div className="scale-plot">
                <div className="scale-gridlines" />
                {scaleRows.map((row) => {
                  return (
                    <button
                      key={row.agents}
                      className={`scale-column ${
                        agentCount === row.agents ? "active" : ""
                      }`}
                      onClick={() => setAgentCount(row.agents)}
                    >
                      <span className="scale-bars">
                        <i
                          className="throughput-bar"
                          style={{
                            height: `${Math.max(5, (row.throughput / 5) * 100)}%`,
                          }}
                        >
                          <b>{row.throughput.toFixed(2)}</b>
                        </i>
                        <i
                          className="team-reward-bar"
                          style={{
                            height: `${Math.max(5, (row.reward / 300) * 100)}%`,
                          }}
                        >
                          <b>{row.reward}</b>
                        </i>
                      </span>
                      <span className="scale-column-label">
                        <strong>{row.agents}</strong>
                        <small>主体</small>
                      </span>
                      <em>{row.stage}</em>
                    </button>
                  );
                })}
              </div>
              <div className="scale-axis right-axis">
                <strong>团队总奖励</strong>
                <span>300</span>
                <span>225</span>
                <span>150</span>
                <span>75</span>
                <span>0</span>
              </div>
            </div>

            <div className="effect-table">
              <div className="effect-table-head">
                <span>规模</span>
                <span>送餐 / 180s</span>
                <span>送餐 / min</span>
                <span>团队总奖励</span>
                <span>协同增益</span>
                <span>边际吞吐</span>
                <span>阶段判断</span>
              </div>
              {scaleRows.map((row) => (
                <button
                  className={agentCount === row.agents ? "active" : ""}
                  key={`effect-${row.agents}`}
                  onClick={() => setAgentCount(row.agents)}
                >
                  <span>
                    <b>{row.agents}</b> 主体
                  </span>
                  <strong>{row.orders} 份</strong>
                  <strong>{row.throughput.toFixed(2)}</strong>
                  <strong>{row.reward}</strong>
                  <strong
                    className={
                      row.synergy < 0 ? "negative-effect" : "positive-effect"
                    }
                  >
                    {row.agents === 1 ? "基线" : signedPercent(row.synergy)}
                  </strong>
                  <strong
                    className={
                      row.marginal !== null && row.marginal <= 0
                        ? "negative-effect"
                        : ""
                    }
                  >
                    {row.marginal === null ? "—" : signedPercent(row.marginal)}
                  </strong>
                  <span className={`stage-tag stage-${row.agents}`}>
                    {row.stage}
                  </span>
                </button>
              ))}
              <p>
                协同增益 = 实际吞吐相对“N 个互不协作的单主体线性基线”的提升；边际吞吐 =
                新增第 N 个主体后，相对 N−1 主体的吞吐变化。
              </p>
            </div>
          </div>

          <div className="metrics-card panel">
            <div className="metrics-heading">
              <span>CORE METRICS</span>
              <strong>{agentCount} 主体评测切片</strong>
            </div>
            <div className="metric-grid">
              <div>
                <span>平均订单时延</span>
                <strong>{benchmarks[agentCount - 1].latency}s</strong>
                <small>↓ 越低越好</small>
              </div>
              <div>
                <span>空闲占比</span>
                <strong>{benchmarks[agentCount - 1].idle}%</strong>
                <small>执行资源浪费</small>
              </div>
              <div>
                <span>交接成功率</span>
                <strong>
                  {agentCount === 1
                    ? "N/A"
                    : `${benchmarks[agentCount - 1].handoff}%`}
                </strong>
                <small>窗口内无掉落</small>
              </div>
              <div>
                <span>碰撞 / 百步</span>
                <strong>{benchmarks[agentCount - 1].collision}</strong>
                <small>拥堵与死锁代理</small>
              </div>
            </div>
          </div>

          <div className="reward-card panel">
            <div className="metrics-heading">
              <span>REWARD DESIGN</span>
              <strong>团队奖励 + 势函数塑形</strong>
            </div>
            <div className="reward-table">
              {rewardRows.map(([name, value, scope]) => (
                <div key={name}>
                  <span>{name}</span>
                  <strong className={value.startsWith("−") ? "negative" : ""}>
                    {value}
                  </strong>
                  <small>{scope}</small>
                </div>
              ))}
            </div>
            <p>
              奖励只由权威环境状态机产生的事件结算，网页不能修改状态或分数。势函数仅使用
              γΦ(s′)−Φ(s) 的转移差分，非法或重复交互不产生奖励。
            </p>
          </div>
        </div>
      </section>

      <section className="closed-loop-section" id="closed-loop">
        <div className="loop-copy">
          <span className="eyebrow">SIM-TO-REAL-TO-SIM</span>
          <h2>把失败变成下一轮训练数据</h2>
          <p>
            真机侧只上传脱敏的失败片段与事件标签；仿真侧进行参数扰动、场景扩增和优先经验回放，
            形成可规模化的反向学习闭环。
          </p>
          <button className="primary-button" onClick={() => setConfigOpen(true)}>
            查看闭环训练配置
          </button>
        </div>
        <div className="loop-flow">
          <article>
            <span className="flow-index">01</span>
            <i className="flow-icon">⌖</i>
            <div>
              <strong>真机部署</strong>
              <small>Decentralized execution</small>
            </div>
            <b>局部感知策略</b>
          </article>
          <span className="flow-arrow">→</span>
          <article>
            <span className="flow-index">02</span>
            <i className="flow-icon alert">REC</i>
            <div>
              <strong>失败捕获</strong>
              <small>Failure case recorder</small>
            </div>
            <b>
              {BASELINE_RECOVERED_FAULTS +
                (motionState.firesExtinguished ?? 0) +
                36}{" "}
              个场景簇
            </b>
          </article>
          <span className="flow-arrow">→</span>
          <article>
            <span className="flow-index">03</span>
            <i className="flow-icon">∞</i>
            <div>
              <strong>规模化变体</strong>
              <small>Domain randomization</small>
            </div>
            <b>× 64 参数扰动</b>
          </article>
          <span className="flow-arrow">→</span>
          <article>
            <span className="flow-index">04</span>
            <i className="flow-icon">∆</i>
            <div>
              <strong>MAPPO 再训练</strong>
              <small>Centralized critic</small>
            </div>
            <b>优先回放 30%</b>
          </article>
        </div>
      </section>

      <footer>
        <div className="footer-brand">
          <span className="brand-mark small">
            <i />
            <i />
            <i />
          </span>
          <span>NEXUS Multi-Agent Emergence Lab</span>
        </div>
        <span>Official-gameplay-derived standalone Burger MARL core</span>
        <span>CTDE / MAPPO / Sim-to-Real</span>
      </footer>

      {configOpen && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setConfigOpen(false);
          }}
        >
          <section
            className="config-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="config-title"
          >
            <div className="modal-head">
              <div>
                <span className="eyebrow">TRAINING BLUEPRINT</span>
                <h2 id="config-title">MAPPO / CTDE 配置</h2>
              </div>
              <button onClick={() => setConfigOpen(false)} aria-label="关闭">
                ×
              </button>
            </div>
            <div className="architecture-strip">
              <div>
                <span>LOCAL OBS</span>
                <strong>9 × 9 语义栅格</strong>
                <small>朝向 · 手持物 · 局部订单</small>
              </div>
              <b>→</b>
              <div>
                <span>SHARED ACTOR</span>
                <strong>CNN + GRU 128</strong>
                <small>主体 ID / 角色嵌入</small>
              </div>
              <b>→</b>
              <div>
                <span>LOCAL ACTION</span>
                <strong>6 离散动作</strong>
                <small>移动 · 交互 · 停留</small>
              </div>
            </div>
            <div className="critic-card">
              <span>CENTRALIZED CRITIC · 仅训练时可见</span>
              <div>
                <strong>全局状态</strong>
                <i>+</i>
                <strong>主体掩码</strong>
                <i>+</i>
                <strong>订单队列</strong>
                <i>+</i>
                <strong>故障状态</strong>
              </div>
            </div>
            <div className="config-grid">
              <div>
                <span>优化器</span>
                <strong>Adam · 3e−4</strong>
              </div>
              <div>
                <span>折扣 / GAE</span>
                <strong>γ 0.99 · λ 0.95</strong>
              </div>
              <div>
                <span>PPO clip</span>
                <strong>ε 0.20</strong>
              </div>
              <div>
                <span>熵系数</span>
                <strong>0.01 → 0.001</strong>
              </div>
              <div>
                <span>并行环境</span>
                <strong>64 × 180 秒</strong>
              </div>
              <div>
                <span>主体数课程</span>
                <strong>1 → 2 → 3 → 4</strong>
              </div>
            </div>
            <div className="modal-note">
              <span>说明</span>
              <p>
                单主体视图回放当前已部署的 PPO checkpoint；2–4 主体仍是确定性规则演示，
                不是 MAPPO 训练结果。正式训练只允许 standalone BurgerEnv
                裁决移动、相邻且无需朝向的上下文动作、原子取放、熟牛肉必须用餐盘承接、计时、物料守恒和团队奖励；网页无权改写状态或计分。
              </p>
            </div>
            <div className="modal-actions">
              <button className="ghost-button" onClick={exportReplay}>
                下载回放 Schema
              </button>
              <button className="primary-button" onClick={() => setConfigOpen(false)}>
                返回仿真
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
