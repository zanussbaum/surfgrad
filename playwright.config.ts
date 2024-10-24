import { PlaywrightTestConfig } from "@playwright/test";

const config: PlaywrightTestConfig = {
  use: {
    headless: true,
    launchOptions: {
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan,UseSkiaRenderer",
        "--enable-dawn-features=allow_unsafe_apis"
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
  projects: [
    {
      name: 'chromium',
      use: { 
        browserName: 'chromium',
        // Use Chrome Canary for better WebGPU support
        channel: 'chrome'
      },
    }
  ]
};

export default config;