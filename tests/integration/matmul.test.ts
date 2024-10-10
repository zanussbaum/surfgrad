import { test, expect } from "@playwright/test";

test("MatMul forward and backward pass", async ({ page }) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  // Inject your test function
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-ignore
      import("/dist/bundle.js").then((module) => {
        const { Tensor } = module;
        // @ts-ignore
        window.runMatMulTest = async function () {
          const x = new Tensor(
            new Float32Array([1, 2, 3, 4, 5, 6]),
            [3, 2],
            true,
          );
          const w = new Tensor(
            new Float32Array([0.1, 0.2, 0.3, 0.4]),
            [2, 2],
            true,
          );

          // Forward pass
          const [y, _] = await x.matmul(w);

          const loss = new Tensor(
            new Float32Array(y.data.length).fill(1),
            y.shape,
            true,
          );

          await y.backward();

          return {
            x: x,
            w: w,
            y: y,
            loss: loss,
            grad_x: x.grad,
            grad_w: w.grad,
          };
        };
        resolve();
      });
    });
  });

  // Run the test function in the browser context
  // @ts-ignore
  const result = await page.evaluate(() => window.runMatMulTest());

  // Perform assertions
  expect(result.x.shape).toEqual([3, 2]);
  expect(result.w.shape).toEqual([2, 2]);
  expect(result.y.shape).toEqual([3, 2]);
  expect(result.grad_x.shape).toEqual([3, 2]);
  expect(result.grad_w.shape).toEqual([2, 2]);

  const gradXData = new Float32Array(Object.values(result.grad_x.data));
  const gradWData = new Float32Array(Object.values(result.grad_w.data));
  const yData = new Float32Array(Object.values(result.y.data));

  expect(yData).toEqual(
    new Float32Array([
      0.7000000476837158, 1, 1.5, 2.200000047683716, 2.3000001907348633,
      3.4000000953674316,
    ]),
  );

  expect(gradXData).toEqual(
    new Float32Array([
      0.30000001192092896, 0.7000000476837158, 0.30000001192092896,
      0.7000000476837158, 0.30000001192092896, 0.7000000476837158,
    ]),
  );

  expect(gradWData).toEqual(new Float32Array([9, 9, 12, 12]));

  await page.close();
});
