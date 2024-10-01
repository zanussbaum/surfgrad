import { PlaywrightTestConfig } from "@playwright/test";

const config: PlaywrightTestConfig = {
  use: {
    headless: true,
  },
  testDir: "./tests/integration",
  webServer: {
    command: "npm run dev",
    url: "http://localhost:8080", // Adjust this if your dev server uses a different port
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000, // Increase timeout if your build process takes longer
  },
};

export default config;
