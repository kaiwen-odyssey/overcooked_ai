import assert from "node:assert/strict";
import { spawn } from "node:child_process";

const port = Number.parseInt(process.env.BURGER_WEBUI_SMOKE_PORT ?? "4173", 10);
const origin = `http://127.0.0.1:${port}`;
const server = spawn(process.execPath, ["dist/standalone/server.js"], {
  cwd: new URL("..", import.meta.url),
  env: {
    ...process.env,
    HOST: "127.0.0.1",
    PORT: String(port),
  },
  stdio: ["ignore", "pipe", "pipe"],
});

let output = "";
server.stdout.on("data", (chunk) => {
  output += chunk.toString();
});
server.stderr.on("data", (chunk) => {
  output += chunk.toString();
});

async function waitForServer() {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (server.exitCode !== null) {
      throw new Error(`WebUI exited before startup:\n${output}`);
    }
    try {
      const response = await fetch(origin);
      if (response.ok) return response;
    } catch {
      // The production server is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`Timed out waiting for ${origin}:\n${output}`);
}

try {
  const response = await waitForServer();
  const html = await response.text();
  assert.match(html, /NEXUS/);
  assert.match(html, /汉堡协作厨房/);

  const artifactResponse = await fetch(
    `${origin}/ppo-single-agent-artifact.json`,
  );
  assert.equal(artifactResponse.status, 200);
  const artifact = await artifactResponse.json();
  assert.equal(artifact.acceptance.accepted, true);
  assert.equal(artifact.evaluation.episodes, 160);
  console.log(`Runnable Burger WebUI verified at ${origin}`);
} finally {
  server.kill("SIGTERM");
  await new Promise((resolve) => {
    const timeout = setTimeout(resolve, 2_000);
    server.once("exit", () => {
      clearTimeout(timeout);
      resolve();
    });
  });
  if (server.exitCode === null) server.kill("SIGKILL");
}
