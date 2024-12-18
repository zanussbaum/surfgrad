import { test, expect } from "@playwright/test";

test("Linear forward pass with known values", async ({ page }) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  // Inject test function
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-expect-error ignore error for tests
      import("/dist/bundle.js").then((module) => {
        const { Tensor, Linear } = module;

        window.runLinearTest = async function () {
          const inputSize = 3;
          const outputSize = 2;

          // Create linear layer
          const linear = new Linear(inputSize, outputSize);

          // Set known weights and biases for deterministic testing
          linear.weight.data = new Float32Array([
            0.1,
            0.2, // First row
            0.3,
            0.4, // Second row
            0.5,
            0.6, // Third row
          ]);

          linear.bias.data = new Float32Array([0.1, 0.2]);

          // Create input tensor
          const input = new Tensor(
            new Float32Array([1.0, 2.0, 3.0]), // Sample input
            [1, 3], // Batch size 1, input size 3
            false,
          );

          // Forward pass
          const [output] = await linear.forward(input);

          return {
            inputData: Array.from(input.data),
            weights: Array.from(linear.weight.data),
            biases: Array.from(linear.bias.data),
            outputShape: output.shape,
            outputData: Array.from(output.data),
          };
        };
        resolve();
      });
    });
  });

  // Run the test function in the browser context
  const result = await page.evaluate(() => window.runLinearTest());

  // Validate shapes
  expect(result.outputShape).toEqual([1, 2]); // Batch size x Output size

  // Calculate expected output manually:
  // output[0] = (1.0 * 0.1 + 2.0 * 0.3 + 3.0 * 0.5) + 0.1 = 2.0
  // output[1] = (1.0 * 0.2 + 2.0 * 0.4 + 3.0 * 0.6) + 0.2 = 2.8
  const expectedOutput = [2.3, 3.0];

  // Check if outputs match expected values within a small tolerance
  expect(result.outputData[0]).toBeCloseTo(expectedOutput[0], 5);
  expect(result.outputData[1]).toBeCloseTo(expectedOutput[1], 5);

  await page.close();
});

declare global {
  interface Window {
    runLinearTest: () => Promise<{
      inputData: number[];
      weights: number[];
      biases: number[];
      outputShape: number[];
      outputData: number[];
    }>;
  }
}
