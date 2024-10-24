import { test, expect } from "@playwright/test";

test("Elementwise ln forward and backward pass", async ({ page }) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  // Inject your test function
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-expect-error ignore error for tests
      import("/dist/bundle.js").then((module) => {
        const { Tensor } = module;

        // @ts-expect-error ignore error for tests
        window.runMulTest = async function () {
          const x = new Tensor(
            new Float32Array([1, 2, 3, 4, 5, 6]),
            [2, 3],
            true,
          );

          // Forward pass
          const [z, _] = await x.ln();

          await z.backward();

          return {
            x: x,
            z: z,
            grad_x: x.grad,
          };
        };
        resolve();
      });
    });
  });

  // Run the test function in the browser context
  // @ts-expect-error ignore error for tests
  const result = await page.evaluate(() => window.runMulTest());

  // Perform assertions
  expect(result.x.shape).toEqual([2, 3]);
  expect(result.z.shape).toEqual([2, 3]);
  expect(result.grad_x.shape).toEqual([2, 3]);

  const zData = new Float32Array(Object.values(result.z.data));
  const gradXData = new Float32Array(Object.values(result.grad_x.data));

  expect(zData).toEqual(
    new Float32Array([
      0.0, 0.6931471824645996, 1.0986121892929077, 1.3862943649291992,
      1.6094378232955933, 1.7917593717575073,
    ]),
  );

  expect(gradXData).toEqual(
    new Float32Array([
      1.0, 0.5, 0.3333333432674408, 0.25, 0.2, 0.1666666716337204,
    ]),
  );

  await page.close();
});
