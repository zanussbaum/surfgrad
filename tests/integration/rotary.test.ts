import { test, expect } from "@playwright/test";

test("Rotary positional embedding forward pass with known values", async ({
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
        const { Tensor, RotaryEmbedding } = module;

        window.runRotaryEmbeddingTest = async function () {
          const seqLength = 4;
          const dimension = 8; // Must be divisible by 2 for rotary embeddings
          const base = 10000.0;

          // Create rotary embedding layer
          const rotaryEmbed = new RotaryEmbedding(base, dimension);

          // Create sample input tensor
          const input = new Tensor(
            new Float32Array([
              0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6,
              0.7, 0.8, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.4, 0.5,
              0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
            ]),
            [seqLength, dimension],
            false,
          );

          // Forward pass
          const [rotatedOutput] = await rotaryEmbed.forward(input);
          console.log("rotatedOutput", rotatedOutput.data.toString());

          // Get the theta values and position encodings for verification
          const theta = rotaryEmbed.createTheta(dimension, base);
          const [cache] = await rotaryEmbed.buildCache(seqLength);

          return {
            inputShape: input.shape,
            inputData: Array.from(input.data),
            outputShape: rotatedOutput.shape,
            outputData: Array.from(rotatedOutput.data),
            theta: Array.from(theta),
            cache: Array.from(cache.data),
            idxTheta: Array.from(rotaryEmbed.idxTheta),
          };
        };
        resolve();
      });
    });
  });

  // Run the test function in the browser context
  const result = await page.evaluate(() => window.runRotaryEmbeddingTest());

  // Validate shapes
  expect(result.inputShape).toEqual([4, 8]);
  expect(result.outputShape).toEqual([4, 8]);

  // Validate theta calculation
  const expectedTheta = Array.from([1.0, 0.1, 0.01, 0.001]);

  result.theta.forEach((value, idx) => {
    expect(value).toBeCloseTo(expectedTheta[idx], 4);
  });

  const expectedIdxTheta = Array.from([
    0.0, 0.0, 0.0, 0.0, 1.0, 1.0e-1, 1.0e-2, 1.0e-3, 2.0, 2.0e-1, 2.0e-2,
    2.0e-3, 3.0, 3.0e-1, 3.0e-2, 3.0e-3,
  ]);

  result.idxTheta.forEach((value, idx) => {
    expect(value).toBeCloseTo(expectedIdxTheta[idx], 4);
  });

  // Validate cache calculation
  const expectedCacheOutput = Array.from([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0,
    1.0e-1, 1.0e-2, 1.0e-3, 2.0, 2.0e-1, 2.0e-2, 2.0e-3, 2.0, 2.0e-1, 2.0e-2,
    2.0e-3, 3.0, 3.0e-1, 3.0e-2, 3.0e-3, 3.0, 3.0e-1, 3.0e-2, 3.0e-3,
  ]);

  result.cache.forEach((value, idx) => {
    expect(value).toBeCloseTo(expectedCacheOutput[idx], 4);
  });

  const expectedRotatedOutput = Array.from([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.3968, 0.2286, 0.392, 0.4991,
    0.4925, 0.7265, 0.804, 0.9005, -0.7614, 0.2331, 0.4819, 0.598, -0.0185,
    0.8635, 0.9098, 1.0012, -0.5089, 0.2117, 0.5697, 0.6967, -0.7355, 1.0076,
    1.0175, 1.1021,
  ]);

  result.outputData.forEach((value, idx) => {
    expect(value).toBeCloseTo(expectedRotatedOutput[idx], 4);
  });

  await page.close();
});

declare global {
  interface Window {
    runRotaryEmbeddingTest: () => Promise<{
      inputShape: number[];
      inputData: number[];
      outputShape: number[];
      outputData: number[];
      theta: number[];
      cache: number[];
      idxTheta: number[];
    }>;
  }
}
