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

  const mulResultData = new Float32Array(Object.values(result.mulResult.data));
  const expResultData = new Float32Array(Object.values(result.expResult.data));
  const addResultData = new Float32Array(Object.values(result.addResult.data));
  const lnResultData = new Float32Array(Object.values(result.lnResult.data));
  const reluResultData = new Float32Array(
    Object.values(result.reluResult.data),
  );
  const addtwoResultData = new Float32Array(
    Object.values(result.addtwoResult.data),
  );
  const outputData = new Float32Array(Object.values(result.output.data));

  expect(result.mulResult.shape).toEqual([1, 1]);
  expect(result.expResult.shape).toEqual([1, 1]);

  expect(mulResultData).toEqual(new Float32Array([6.0]));
  expect(expResultData).toEqual(new Float32Array([403.4287109375]));
  expect(addResultData).toEqual(new Float32Array([404.4287109375]));
  expect(lnResultData).toEqual(new Float32Array([6.002475261688232]));
  expect(reluResultData).toEqual(new Float32Array([6.002475261688232]));
  expect(addtwoResultData).toEqual(new Float32Array([8.002475261688232]));
  expect(outputData).toEqual(new Float32Array([2.0797510147094727]));

  const addtwoResultGradData = new Float32Array(
    Object.values(result.addtwoResult.grad.data),
  );
  const reluResultGradData = new Float32Array(
    Object.values(result.reluResult.grad.data),
  );
  const lnResultGradData = new Float32Array(
    Object.values(result.lnResult.grad.data),
  );
  const addResultGradData = new Float32Array(
    Object.values(result.addResult.grad.data),
  );
  const expResultGradData = new Float32Array(
    Object.values(result.expResult.grad.data),
  );
  const mulResultGradData = new Float32Array(
    Object.values(result.mulResult.grad.data),
  );
  const xGradData = new Float32Array(Object.values(result.x.grad.data));
  const yGradData = new Float32Array(Object.values(result.y.grad.data));
  const outputGradData = new Float32Array(
    Object.values(result.output.grad.data),
  );

  expect(outputGradData).toEqual(new Float32Array([1]));
  expect(addtwoResultGradData).toEqual(new Float32Array([0.12496133148670197]));
  expect(reluResultGradData).toEqual(new Float32Array([0.12496133148670197]));
  expect(lnResultGradData).toEqual(new Float32Array([0.12496133148670197]));
  expect(addResultGradData).toEqual(new Float32Array([0.00030898235854692757]));
  expect(expResultGradData).toEqual(new Float32Array([0.00030898235854692757]));
  expect(mulResultGradData).toEqual(new Float32Array([0.12465235590934753]));
  expect(xGradData).toEqual(new Float32Array([0.3739570677280426]));
  expect(yGradData).toEqual(new Float32Array([0.24930469691753387]));

  await page.close();
});
