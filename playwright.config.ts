import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  // Cap workers to avoid resource contention from many simultaneous Electron instances.
  // Note: workers is a top-level-only option — per-project workers is silently ignored.
  // Use the same cap locally and in CI; the Electron suites are I/O heavy and become flaky
  // when too many app instances boot at once.
  workers: 4,
  reporter: process.env.CI ? [["github"], ["html"]] : "html",
  timeout: 60000,
  use: {
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "unit",
      testDir: "./tests/unit",
      testMatch: /.*\.spec\.ts/,
      fullyParallel: true,
    },
    {
      name: "e2e",
      testDir: "./tests/e2e",
      testMatch: /.*\.spec\.ts/,
      // Each worker gets an isolated database via TEST_WORKER_INDEX,
      // so E2E tests can now run fully in parallel across files.
      // Tests within a describe block stay serial (they share an Electron instance).
      fullyParallel: true,
    },
    {
      name: "integration",
      testDir: "./tests",
      testMatch: /.*\.spec\.ts/,
      testIgnore: [/unit\//, /e2e\//, /problematic\//],
      fullyParallel: true,
    },
    {
      name: "problematic",
      testDir: "./tests/problematic",
      testMatch: /.*\.spec\.ts/,
      // These tests are flaky and excluded from the main test run
      // Run manually with: npx playwright test --project=problematic
      fullyParallel: false,
      workers: 1,
    },
    {
      name: "benchmark",
      testDir: "./benchmarks",
      testMatch: /.*\.spec\.ts/,
      // Performance benchmarks — not part of CI, run manually with:
      // npx playwright test --project=benchmark
      fullyParallel: false,
      workers: 1,
    },
  ],
});
