import { test, expect } from "@playwright/test";

test("Elementwise scalar/broadcasted multiplication forward and backward pass", async ({
  page,
}) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  // Inject your test function
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-ignore
      import("/dist/bundle.js").then((module) => {
        const { Tensor, Mul } = module;

        // @ts-ignore
        window.runMulTest = async function () {
          const x = new Tensor(
            new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [2, 3],
            true,
          );
          const y = new Tensor(new Float32Array([2.0]), [1], false);

          let operation = await Mul.create();

          // Forward pass
          const z = await operation.forward(x, y);

          const loss = new Tensor(new Float32Array(z.data), z.shape, true);

          // Backward pass
          const [grad_x, grad_y] = await operation.backward(loss);

          return {
            x: x,
            y: y,
            z: z,
            loss: loss,
            grad_x: grad_x,
            grad_y: grad_y,
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
  expect(result.y.shape).toEqual([1]);
  expect(result.z.shape).toEqual([2, 3]);
  expect(result.grad_x.shape).toEqual([2, 3]);
  // check that grad_y is undefined
  expect(result.grad_y).toBeUndefined();

  const zData = new Float32Array(Object.values(result.z.data));
  const gradXData = new Float32Array(Object.values(result.grad_x.data));

  expect(zData).toEqual(new Float32Array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]));

  expect(gradXData).toEqual(
    new Float32Array([4.0, 8.0, 12.0, 16.0, 20.0, 24.0]),
  );

  await page.close();
});

test("Elementwise multiplication forward and backward pass", async ({
  page,
}) => {
  await page.goto("http://localhost:8080");

  // Inject your test function
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-ignore
      import("/dist/bundle.js").then((module) => {
        const { Tensor, Mul } = module;

        // @ts-ignore
        window.runMulTest = async function () {
          const x = new Tensor(
            new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [2, 3],
            true,
          );
          const y = new Tensor(
            new Float32Array([2.0, 0.5, 2.0, 0.5, 2.0, 0.5]),
            [2, 3],
            true,
          );

          let operation = await Mul.create();

          // Forward pass
          const z = await operation.forward(x, y);

          const loss = new Tensor(new Float32Array(z.data), z.shape, true);

          // Backward pass
          const [grad_x, grad_y] = await operation.backward(loss);

          return {
            x: x,
            y: y,
            z: z,
            loss: loss,
            grad_x: grad_x,
            grad_y: grad_y,
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
  expect(result.y.shape).toEqual([2, 3]);
  expect(result.z.shape).toEqual([2, 3]);
  expect(result.grad_x.shape).toEqual([2, 3]);
  expect(result.grad_y.shape).toEqual([2, 3]);

  const zData = new Float32Array(Object.values(result.z.data));
  const gradXData = new Float32Array(Object.values(result.grad_x.data));
  const gradYData = new Float32Array(Object.values(result.grad_y.data));

  expect(zData).toEqual(new Float32Array([2.0, 1.0, 6.0, 2.0, 10.0, 3.0]));

  expect(gradXData).toEqual(new Float32Array([4.0, 0.5, 12.0, 1.0, 20.0, 1.5]));

  expect(gradYData).toEqual(
    new Float32Array([2.0, 2.0, 18.0, 8.0, 50.0, 18.0]),
  );

  await page.close();
});
