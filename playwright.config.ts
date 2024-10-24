import { PlaywrightTestConfig } from "@playwright/test";

const config: PlaywrightTestConfig = {
  use: {
    headless: true,
    launchOptions: {
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan",
        "--enable-dawn-features=allow_unsafe_apis",
        "--disable-dawn-features=disallow_unsafe_apis",
        // Adding these based on common WebGPU flags
        "--enable-gpu",
        "--no-sandbox",
        "--ignore-gpu-blacklist",
      ]
    }
  },
  testDir: "./tests/integration",
  webServer: {
    command: "npm run dev",
    url: "http://localhost:8080",
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
};

export default config;