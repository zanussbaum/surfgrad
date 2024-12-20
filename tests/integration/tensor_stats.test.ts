import { Tensor } from "../../src/tensor/tensor.js";
import { test, expect } from "@playwright/test";

test.describe("Tensor Statistics Operations", () => {
  test.describe("mean", () => {
    test("should calculate mean along specified dimensions", async () => {
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const tensor = new Tensor(data, [2, 3], false);

      const mean = await tensor.mean([1]);
      expect(mean.shape).toEqual([2]);
      expect(Array.from(mean.data)).toEqual([2, 5]); // [mean(1,2,3), mean(4,5,6)]
    });

    test("should handle single dimension tensors", async () => {
      const data = new Float32Array([1, 2, 3, 4]);
      const tensor = new Tensor(data, [4], false);

      const mean = await tensor.mean([0]);
      expect(mean.shape).toEqual([1]);
      expect(mean.data[0]).toBeCloseTo(2.5); // mean(1,2,3,4)
    });
  });

  test.describe("variance", () => {
    test("should calculate variance along specified dimensions", async () => {
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const tensor = new Tensor(data, [2, 3], false);

      const variance = await tensor.variance([1]);
      expect(variance.shape).toEqual([2]);
      // Variance of [1,2,3] and [4,5,6]
      expect(
        Array.from(variance.data).map((x) => Number(x.toFixed(2))),
      ).toEqual([0.67, 0.67]);
    });

    test("should handle single dimension tensors", async () => {
      const data = new Float32Array([2, 4, 4, 6]);
      const tensor = new Tensor(data, [4], false);

      const variance = await tensor.variance([0]);
      expect(variance.shape).toEqual([1]);
      expect(variance.data[0]).toBeCloseTo(2); // variance of [2,4,4,6]
    });
  });

  test.describe("sqrt", () => {
    test("should calculate element-wise square root", async () => {
      const data = new Float32Array([1, 4, 9, 16]);
      const tensor = new Tensor(data, [4], false);

      const sqrt = await tensor.sqrt();
      expect(sqrt.shape).toEqual([4]);
      expect(Array.from(sqrt.data)).toEqual([1, 2, 3, 4]);
    });

    test("should handle multi-dimensional tensors", async () => {
      const data = new Float32Array([1, 4, 9, 16, 25, 36]);
      const tensor = new Tensor(data, [2, 3], false);

      const sqrt = await tensor.sqrt();
      expect(sqrt.shape).toEqual([2, 3]);
      expect(Array.from(sqrt.data)).toEqual([1, 2, 3, 4, 5, 6]);
    });
  });

  test.describe("combined operations", () => {
    test("should correctly compute standard deviation using sqrt(variance)", async () => {
      const data = new Float32Array([2, 4, 4, 6]);
      const tensor = new Tensor(data, [4], false);

      const variance = await tensor.variance([0]);
      const stdDev = await variance.sqrt();

      expect(stdDev.shape).toEqual([1]); // The shape should be [1] for a scalar result
      expect(stdDev.data[0]).toBeCloseTo(Math.sqrt(2)); // The actual value check
    });
  });
});
