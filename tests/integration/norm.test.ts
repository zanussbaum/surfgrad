import { test, expect } from "@playwright/test";

test("LayerNorm forward pass with known values", async ({ page }) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  // Inject test function
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-expect-error ignore error for tests
      import("/dist/bundle.js").then((module) => {
        const { Tensor, LayerNorm } = module;

        window.runLayerNormTest = async function () {
          // Create a simple input tensor with known values
          const input = new Tensor(
            new Float32Array([1, 2, 3, 4, 5, 6]), // Sample values
            [2, 3], // 2 sequences, 3 features each
            false,
          );

          // Create LayerNorm with normalized_shape [3]
          const layerNorm = new LayerNorm([3], 1e-5);

          // Set known values for gamma and beta
          layerNorm.gamma.data.set([1.0, 1.0, 1.0]);
          layerNorm.beta.data.set([0.0, 0.0, 0.0]);

          // Forward pass
          const [output] = await layerNorm.forward(input);

          return {
            inputData: Array.from(input.data),
            inputShape: input.shape,
            outputShape: output.shape,
            outputData: Array.from(output.data),
            gamma: Array.from(layerNorm.gamma.data),
            beta: Array.from(layerNorm.beta.data),
          };
        };
        resolve();
      });
    });
  });

  // Run the test function in the browser context
  const result = await page.evaluate(() => window.runLayerNormTest());

  // Validate shapes
  expect(result.inputShape).toEqual([2, 3]);
  expect(result.outputShape).toEqual([2, 3]);

  // For the input [1,2,3] and [4,5,6], with gamma=1 and beta=0,
  // we can pre-calculate the expected normalized values
  const expectedOutput = [
    -1.224744871391589,
    0,
    1.224744871391589, // First sequence normalized
    -1.224744871391589,
    0,
    1.224744871391589, // Second sequence normalized
  ];

  // Check if output matches expected values (using approximate equality)
  result.outputData.forEach((val, idx) => {
    expect(val).toBeCloseTo(expectedOutput[idx], 4);
  });

  await page.close();
});

declare global {
  interface Window {
    runLayerNormTest: () => Promise<{
      inputData: number[];
      inputShape: number[];
      outputShape: number[];
      outputData: number[];
      gamma: number[];
      beta: number[];
    }>;
  }
}
