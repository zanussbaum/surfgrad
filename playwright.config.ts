import { PlaywrightTestConfig } from "@playwright/test";

const config: PlaywrightTestConfig = {
  use: {
    headless: true,
    launchOptions: {
      args: [
        "--enable-features=Vulkan,UseSkiaRenderer",
        "--use-vulkan=swiftshader",
        "--enable-unsafe-webgpu",
        "--disable-vulkan-fallback-to-gl-for-testing",
        // '--dignore-gpu-blocklist',
        "--use-angle=vulkan",
      ],
    },
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
