import { test, expect } from "@playwright/test";

test("MultiHeadAttention forward pass with known values", async ({ page }) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  // Inject test function
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-expect-error ignore error for tests
      import("/dist/bundle.js").then((module) => {
        const { Tensor, MultiHeadAttention } = module;

        window.runAttentionTest = async function () {
          // Create sample input tensor with known values
          const seqLength = 2;
          const hiddenDim = 4;
          const numHeads = 2;

          const input = new Tensor(
            new Float32Array([
              0.1,
              0.2,
              0.3,
              0.4, // First sequence
              0.5,
              0.6,
              0.7,
              0.8, // Second sequence
            ]),
            [seqLength, hiddenDim],
            false,
          );

          // Create MultiHeadAttention
          const attention = new MultiHeadAttention(hiddenDim, numHeads);

          // Set known weights and biases for reproducibility
          attention.qkv.weight = new Tensor(
            new Float32Array([
              // Q weights
              0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4,
              0.5, 0.6, 0.7,
              // K weights
              0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4,
              0.4, 0.4, 0.4,
              // V weights
              0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.8,
              0.8, 0.8, 0.8,
            ]),
            [hiddenDim, hiddenDim * 3],
            true,
          );

          attention.qkv.bias = new Tensor(
            new Float32Array([
              // Q bias
              0.1, 0.1, 0.1, 0.1,
              // K bias
              0.2, 0.2, 0.2, 0.2,
              // V bias
              0.3, 0.3, 0.3, 0.3,
            ]),
            [hiddenDim * 3],
            true,
          );

          attention.output.weight = new Tensor(
            new Float32Array([
              0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4,
              0.5, 0.6, 0.7,
            ]),
            [hiddenDim, hiddenDim],
            true,
          );

          attention.output.bias = new Tensor(
            new Float32Array([0.1, 0.1, 0.1, 0.1]),
            [hiddenDim],
            true,
          );

          // Forward pass
          const [output] = await attention.forward(input);

          return {
            inputShape: input.shape,
            inputData: Array.from(input.data),
            outputShape: output.shape,
            outputData: Array.from(output.data),
          };
        };
        resolve();
      });
    });
  });

  // Run the test function in the browser context
  const result = await page.evaluate(() => window.runAttentionTest());

  // Validate shapes
  expect(result.inputShape).toEqual([2, 4]); // [seq_len, hidden_dim]
  expect(result.outputShape).toEqual([2, 4]); // [seq_len, hidden_dim]
  console.log("result.outputData:", result.outputData.toString());

  // Expected values computed using the same architecture with PyTorch
  const expectedOutput = [
    1.4622, 1.9985, 2.5347, 3.0709, 1.5701, 2.1462, 2.7224, 3.2985,
  ];

  // Validate output values
  result.outputData.forEach((value, idx) => {
    expect(value).toBeCloseTo(expectedOutput[idx], 4);
  });

  await page.close();
});

declare global {
  interface Window {
    runAttentionTest: () => Promise<{
      inputShape: number[];
      inputData: number[];
      outputShape: number[];
      outputData: number[];
    }>;
  }
}
