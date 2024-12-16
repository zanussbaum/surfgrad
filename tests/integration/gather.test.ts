import { test, expect } from "@playwright/test";

test("Gather forward and backward pass", async ({ page }) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-expect-error ignore error for tests
      import("/dist/bundle.js").then((module) => {
        const { Tensor } = module;

        // @ts-expect-error ignore error for tests
        window.runGatherTest = async function () {
          // Create a simple embedding matrix with 3 embeddings of dimension 2
          const embeddings = new Tensor(
            new Float32Array([
              1.0, 2.0,  // embedding 0
              3.0, 4.0,  // embedding 1
              5.0, 6.0   // embedding 2
            ]),
            [3, 2],
            true
          );

          // Look up embeddings at indices 1, 0 (second embedding, then first)
          const indices = new Tensor(
            new Float32Array([1, 0]),
            [2, 1],
            false
          );

          // Forward pass - gather embeddings
          const [output] = await embeddings.gather(indices);

          // Backward pass
          await output.backward();

          return {
            embeddings: embeddings,
            indices: indices,
            output: output,
            grad_embeddings: embeddings.grad,
          };
        };
        resolve();
      });
    });
  });

  // Run the test function in the browser context
  // @ts-expect-error ignore error for tests
  const result = await page.evaluate(() => window.runGatherTest());

  expect(result.output.shape).toEqual([2, 2]);  // 2 selected embeddings of dimension 2
  expect(result.grad_embeddings.shape).toEqual([3, 2]);  // Same shape as input embeddings

  // Forward pass assertions - should get embeddings at indices 1 and 0
  const outputData = new Float32Array(Object.values(result.output.data));
  expect(outputData).toEqual(new Float32Array([
    3.0, 4.0,  // embedding at index 1
    1.0, 2.0   // embedding at index 0
  ]));

  // Backward pass assertions - gradient should accumulate at the selected indices
  const gradData = new Float32Array(Object.values(result.grad_embeddings.data));
  expect(gradData).toEqual(new Float32Array([
    1.0, 1.0,  // gradient for embedding 0 (selected second)
    1.0, 1.0,  // gradient for embedding 1 (selected first)
    0.0, 0.0   // gradient for embedding 2 (not selected)
  ]));

  await page.close();
});