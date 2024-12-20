import { test, expect } from "@playwright/test";

test("MLP with SwiGLU activation forward pass with known values", async ({
  page,
}) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  // Inject test function
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-expect-error ignore error for tests
      import("/dist/bundle.js").then((module) => {
        const { Tensor, MLP } = module;

        window.runSwiGLUTest = async function () {
          // Create sample input tensor with known values
          const inputDim = 4;
          const hiddenDim = 8; // Will be doubled internally for SwiGLU
          const seqLength = 2;

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
            [seqLength, inputDim],
            false,
          );

          // Create MLP with SwiGLU activation
          const mlp = new MLP(inputDim, hiddenDim, "swiglu");

          // Set known weights and biases for reproducibility
          mlp.up.weight = new Tensor(
            new Float32Array([
              // First half for gate
              0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6,
              0.7, 0.8, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.4, 0.5,
              0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
              // Second half for value
              0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2,
              0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4,
              0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
            ]),
            [inputDim, hiddenDim * 2],
            true,
          );

          mlp.up.bias = new Tensor(
            new Float32Array([
              // Gate bias
              0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
              // Value bias
              0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
            ]),
            [hiddenDim * 2],
            true,
          );

          mlp.down.weight = new Tensor(
            new Float32Array([
              0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4,
              0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.8, 0.6, 0.7, 0.8, 0.9, 0.7, 0.8,
              0.9, 1.0, 0.8, 0.9, 1.0, 1.1,
            ]),
            [hiddenDim, inputDim],
            true,
          );

          mlp.down.bias = new Tensor(
            new Float32Array([0.1, 0.1, 0.1, 0.1]),
            [inputDim],
            true,
          );

          // Forward pass
          const [output] = await mlp.forward(input);

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
  const result = await page.evaluate(() => window.runSwiGLUTest());

  // Validate shapes
  expect(result.inputShape).toEqual([2, 4]); // [batch_size, input_dim]
  expect(result.outputShape).toEqual([2, 4]); // [batch_size, input_dim]
  console.log("result.outputData:", result.outputData.toString());

  // Expected values computed using the same architecture with PyTorch
  const expectedOutput = [
    0.7809, 0.9126, 1.0443, 1.176, 5.0712, 5.9646, 6.8581, 7.7515,
  ];

  // Validate output values
  result.outputData.forEach((value, idx) => {
    expect(value).toBeCloseTo(expectedOutput[idx], 4);
  });

  await page.close();
});

declare global {
  interface Window {
    runSwiGLUTest: () => Promise<{
      inputShape: number[];
      inputData: number[];
      outputShape: number[];
      outputData: number[];
    }>;
  }
}
