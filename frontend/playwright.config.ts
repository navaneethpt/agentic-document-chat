import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e", workers: 1, timeout: 30000,
  use: { baseURL: "http://127.0.0.1:8100", trace: "retain-on-failure", channel: process.env.PLAYWRIGHT_CHANNEL },
  webServer: {
    command: "../.venv/bin/python -m uvicorn fake_api:app --app-dir ../tests --host 127.0.0.1 --port 8100",
    env: { PYTHONPATH: ".." }, url: "http://127.0.0.1:8100/api/health",
    reuseExistingServer: false,
  },
});
