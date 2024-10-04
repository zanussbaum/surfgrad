import { test, expect } from "@playwright/test";

test("Elementwise log2 forward and backward pass", async ({ page }) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  // Inject your test function
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-ignore
      import("/dist/bundle.js").then((module) => {
        const { Tensor, Log } = module;

        // @ts-ignore
        window.runMulTest = async function () {
          const x = new Tensor(
            new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [2, 3],
            true,
          );

          let operation = await Log.create();

          // Forward pass
          const [z, _] = await operation.forward(x);

          const loss = new Tensor(
            new Float32Array(z.data.length).fill(1),
            z.shape,
            true,
          );

          // Backward pass
          const [grad_x] = await operation.backward(loss);

          return {
            x: x,
            z: z,
            loss: loss,
            grad_x: grad_x,
          };
        };
        resolve();
      });
    });
  });

  // Run the test function in the browser context
  // @ts-ignore
  const result = await page.evaluate(() => window.runMulTest());

  // Perform assertions
  expect(result.x.shape).toEqual([2, 3]);
  expect(result.z.shape).toEqual([2, 3]);
  expect(result.grad_x.shape).toEqual([2, 3]);

  const zData = new Float32Array(Object.values(result.z.data));
  const gradXData = new Float32Array(Object.values(result.grad_x.data));

  expect(zData).toEqual(
    new Float32Array([
      0.0, 1, 1.5849623680114746, 2, 2.321928024291992, 2.5849623680114746,
    ]),
  );

  expect(gradXData).toEqual(
    new Float32Array([
      1.4426950216293335, 0.7213475108146667, 0.48089835047721863,
      0.3606737554073334, 0.28853902220726013, 0.24044917523860931,
    ]),
  );

  await page.close();
});
