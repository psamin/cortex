import { defineConfig } from "vitest/config";

// Unit tests for the pure-TypeScript kinematics core (no DOM, no WebGL).
export default defineConfig({
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
