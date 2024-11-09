import { test, expect } from "@playwright/test";

test("Autograd graph creation test", async ({ page }) => {
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
          const x = new Tensor(new Float32Array([2.0]), [1, 1], true);
          const y = new Tensor(new Float32Array([3.0]), [1, 1], true);

          const [mulResult] = await x.mul(y);
          const [expResult] = await mulResult.exp();
          const [addResult] = await expResult.add(
            new Tensor(new Float32Array([1.0]), [1], false),
          );
          const [lnResult] = await addResult.ln();
          const [reluResult] = await lnResult.relu();
          const [addtwoResult] = await reluResult.add(
            new Tensor(new Float32Array([2.0]), [1], false),
          );
          const [output] = await addtwoResult.ln();

          // populate gradient
          await output.backward();

          return {
            x: x,
            y: y,
            mulResult: mulResult,
            expResult: expResult,
            addResult: addResult,
            lnResult: lnResult,
            reluResult: reluResult,
            addtwoResult: addtwoResult,
            output: output,
          };
        };
        resolve();
      });
    });
  });

  // Run the test function in the browser context
  // @ts-expect-error ignore error for tests
  const result = await page.evaluate(() => window.runMulTest());

  const mulResultData = Number(Object.values(result.mulResult.data));
  const expResultData = Number(Object.values(result.expResult.data));
  const addResultData = Number(Object.values(result.addResult.data));
  const lnResultData = Number(Object.values(result.lnResult.data));
  const reluResultData = Number(Object.values(result.reluResult.data));
  const addtwoResultData = Number(Object.values(result.addtwoResult.data));
  const outputData = Number(Object.values(result.output.data));

  expect(result.mulResult.shape).toEqual([1, 1]);
  expect(result.expResult.shape).toEqual([1, 1]);

  expect(mulResultData).toBeCloseTo(Number(6.0), 5);
  expect(expResultData).toBeCloseTo(Number(403.4287109375), 5);
  expect(addResultData).toBeCloseTo(Number(404.4287109375), 5);
  expect(lnResultData).toBeCloseTo(Number(6.002475261688232), 5);
  expect(reluResultData).toBeCloseTo(Number(6.002475261688232), 5);
  expect(addtwoResultData).toBeCloseTo(Number(8.002475261688232), 5);
  expect(outputData).toBeCloseTo(Number(2.0797510147094727), 5);

  const addtwoResultGradData = Number(
    Object.values(result.addtwoResult.grad.data)[0],
  );
  const reluResultGradData = Number(
    Object.values(result.reluResult.grad.data)[0],
  );
  const lnResultGradData = Number(Object.values(result.lnResult.grad.data)[0]);
  const addResultGradData = Number(
    Object.values(result.addResult.grad.data)[0],
  );
  const expResultGradData = Number(
    Object.values(result.expResult.grad.data)[0],
  );
  const mulResultGradData = Number(
    Object.values(result.mulResult.grad.data)[0],
  );
  const xGradData = Number(Object.values(result.x.grad.data)[0]);
  const yGradData = Number(Object.values(result.y.grad.data)[0]);
  const outputGradData = Number(Object.values(result.output.grad.data)[0]);

  expect(outputGradData).toBeCloseTo(1, 5);
  expect(addtwoResultGradData).toBeCloseTo(0.12496133148670197, 5);
  expect(reluResultGradData).toBeCloseTo(0.12496133148670197, 5);
  expect(lnResultGradData).toBeCloseTo(0.12496133148670197, 5);
  expect(addResultGradData).toBeCloseTo(0.00030898235854692757, 5);
  expect(expResultGradData).toBeCloseTo(0.00030898235854692757, 5);
  expect(mulResultGradData).toBeCloseTo(0.12465235590934753, 5);
  expect(xGradData).toBeCloseTo(0.3739570677280426, 5);
  expect(yGradData).toBeCloseTo(0.24930471181869507, 5);

  await page.close();
});
