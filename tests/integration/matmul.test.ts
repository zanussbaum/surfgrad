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

declare global {
  interface Window {
    runRandomMatMulTest: (size: number) => Promise<{
      webgpuResult: { [key: string]: number };
      naiveResult: number[];
    }>;
  }
}

test("Random MatMul equivalence test", async ({ page }) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  // Inject test functions
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-ignore
      import("/dist/bundle.js").then((module) => {
        const { Tensor } = module;

        // Naive matrix multiplication
        function naiveMatMul(a: number[][], b: number[][]): number[][] {
          const m = a.length;
          const n = b[0].length;
          const p = b.length;
          const result = Array(m)
            .fill(0)
            .map(() => Array(n).fill(0));

          for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
              for (let k = 0; k < p; k++) {
                result[i][j] += a[i][k] * b[k][j];
              }
            }
          }

          return result;
        }

        // Generate random square matrix
        function randomSquareMatrix(size: number): number[][] {
          return Array(size)
            .fill(0)
            .map(() =>
              Array(size)
                .fill(0)
                .map(() => Math.random()),
            );
        }

        // Convert 2D array to 1D array
        function flattenMatrix(matrix: number[][]): number[] {
          return matrix.reduce((acc, row) => acc.concat(row), []);
        }

        // @ts-ignore
        window.runRandomMatMulTest = async function (size: number) {
          const a = randomSquareMatrix(size);
          const b = randomSquareMatrix(size);

          const tensorA = new Tensor(
            new Float32Array(flattenMatrix(a)),
            [size, size],
            true,
          );
          const tensorB = new Tensor(
            new Float32Array(flattenMatrix(b)),
            [size, size],
            true,
          );

          // WebGPU MatMul
          const [webgpuResult, _] = await tensorA.matmul(tensorB);

          // Naive MatMul
          const naiveResult = naiveMatMul(a, b);

          return {
            webgpuResult: webgpuResult.data,
            naiveResult: flattenMatrix(naiveResult),
          };
        };
        resolve();
      });
    });
  });

  const sizes = [2, 4, 8, 16, 32, 64, 128];
  // for larger matrices, we lose precision
  const digitsOfPrecision = [5, 5, 5, 5, 5, 4, 4];
  for (let i = 0; i < sizes.length; i++) {
    const size = sizes[i];
    const digits = digitsOfPrecision[i];
    // Run the test function in the browser context
    const result = await page.evaluate(
      (size) => window.runRandomMatMulTest(size),
      size,
    );

    const webgpuData = new Float32Array(Object.values(result.webgpuResult));
    const naiveData = new Float32Array(result.naiveResult);

    // Check if shapes match
    expect(webgpuData.length).toBe(naiveData.length);

    // Check if values are close (allowing for small floating-point differences)
    for (let i = 0; i < webgpuData.length; i++) {
      expect(webgpuData[i], { message: `at index ${i}` }).toBeCloseTo(
        naiveData[i],
        digits,
      );
    }
  }

  await page.close();
});
