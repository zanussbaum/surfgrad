import { Tensor } from "../../src/tensor/tensor.js";

describe("Tensor", () => {
  describe("constructor", () => {
    it("should create a tensor with the correct properties", () => {
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const tensor = new Tensor(data, shape, true);

      expect(tensor.data).toBe(data);
      expect(tensor.shape).toEqual(shape);
      expect(tensor.requires_grad).toBe(true);
      expect(tensor.grad).toBeNull();
    });

    it("should default requires_grad to false", () => {
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const tensor = new Tensor(data, shape);

      expect(tensor.requires_grad).toBe(false);
    });

    it("should throw an error if the number of elements in data and shape are different", () => {
      const data = new Float32Array([1, 2, 3]);
      const shape = [2, 2];

      expect(() => new Tensor(data, shape)).toThrow("Incompatible shapes");
    });
  });

  describe("transpose", () => {
    it("should correctly transpose a 2x2 matrix", () => {
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const tensor = new Tensor(data, shape);

      const transposed = tensor.transpose();

      expect(transposed.shape).toEqual([2, 2]);
      expect(Array.from(transposed.data)).toEqual([1, 3, 2, 4]);
    });

    it("should correctly transpose a 2x3 matrix", () => {
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const shape = [2, 3];
      const tensor = new Tensor(data, shape);

      const transposed = tensor.transpose();

      expect(transposed.shape).toEqual([3, 2]);
      expect(Array.from(transposed.data)).toEqual([1, 4, 2, 5, 3, 6]);
    });

    it("should preserve requires_grad when transposing", () => {
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [2, 2];
      const tensor = new Tensor(data, shape, true);

      const transposed = tensor.transpose();

      expect(transposed.requires_grad).toBe(true);
    });

    it("should handle 1xN matrices", () => {
      const data = new Float32Array([1, 2, 3]);
      const shape = [1, 3];
      const tensor = new Tensor(data, shape);

      const transposed = tensor.transpose();

      expect(transposed.shape).toEqual([3, 1]);
      expect(Array.from(transposed.data)).toEqual([1, 2, 3]);
    });

    it("should handle Nx1 matrices", () => {
      const data = new Float32Array([1, 2, 3]);
      const shape = [3, 1];
      const tensor = new Tensor(data, shape);

      const transposed = tensor.transpose();

      expect(transposed.shape).toEqual([1, 3]);
      expect(Array.from(transposed.data)).toEqual([1, 2, 3]);
    });
    it("should create ones_like tensor", () => {
      const data = new Float32Array([1, 2, 3]);
      const shape = [3, 1];
      const tensor = new Tensor(data, shape);

      const ones = Tensor.onesLike(tensor);

      expect(ones.data).toEqual(new Float32Array([1, 1, 1]));
    });
    it("should create zeros_like tensor", () => {
      const data = new Float32Array([1, 2, 3]);
      const shape = [3, 1];
      const tensor = new Tensor(data, shape);

      expect(Tensor.zerosLike(tensor).data).toEqual(
        new Float32Array([0, 0, 0]),
      );
    });
  });

  describe("concat", () => {
    it("should concatenate 1D tensors along axis 0", async () => {
      const t1 = new Tensor(new Float32Array([1, 2, 3]), [3]);
      const t2 = new Tensor(new Float32Array([4, 5, 6]), [3]);

      const result = await t1.concat(t2, 0);

      expect(result.shape).toEqual([6]);
      expect(Array.from(result.data)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("should concatenate 2D tensors along axis 0", async () => {
      const t1 = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
      const t2 = new Tensor(new Float32Array([5, 6, 7, 8]), [2, 2]);

      const result = await t1.concat(t2, 0);

      expect(result.shape).toEqual([4, 2]);
      expect(Array.from(result.data)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    });

    it("should concatenate 2D tensors along axis 1", async () => {
      const t1 = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
      const t2 = new Tensor(new Float32Array([5, 6, 7, 8]), [2, 2]);

      const result = await t1.concat(t2, 1);

      expect(result.shape).toEqual([2, 4]);
      expect(Array.from(result.data)).toEqual([1, 2, 5, 6, 3, 4, 7, 8]);
    });

    it("should throw error for invalid axis", async () => {
      const t1 = new Tensor(new Float32Array([1, 2]), [2]);
      const t2 = new Tensor(new Float32Array([3, 4]), [2]);

      await expect(t1.concat(t2, 1)).rejects.toThrow("Invalid axis");
    });

    it("should throw error for shape mismatch", async () => {
      const t1 = new Tensor(new Float32Array([1, 2]), [2]);
      const t2 = new Tensor(new Float32Array([3, 4, 5]), [3]);

      await expect(t1.concat(t2, 0)).rejects.toThrow("Shape mismatch");
    });

    it("should preserve requires_grad", async () => {
      const t1 = new Tensor(new Float32Array([1, 2]), [2], true);
      const t2 = new Tensor(new Float32Array([3, 4]), [2], false);

      const result = await t1.concat(t2, 0);

      expect(result.requires_grad).toBe(true);
    });
  });
  describe("slice", () => {
    it("should slice a 1D tensor with basic indexing", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5]), [5]);
      const result = await tensor.slice([1, 4]);

      expect(result.shape).toEqual([3]);
      expect(Array.from(result.data)).toEqual([2, 3, 4]);
    });

    it("should handle full slice with ':'", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [4]);
      const result = await tensor.slice(":");

      expect(result.shape).toEqual([4]);
      expect(Array.from(result.data)).toEqual([1, 2, 3, 4]);
    });

    it("should slice with step size", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [6]);
      const result = await tensor.slice([null, null, 2]);

      expect(result.shape).toEqual([3]);
      expect(Array.from(result.data)).toEqual([1, 3, 5]);
    });

    it("should handle negative indices", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5]), [5]);
      const result = await tensor.slice([-3, -1]);

      expect(result.shape).toEqual([2]);
      expect(Array.from(result.data)).toEqual([3, 4]);
    });

    it("should slice a 2D tensor along both dimensions", async () => {
      const tensor = new Tensor(
        new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        [3, 3],
      );
      const result = await tensor.slice([0, 2], [1, 3]);

      expect(result.shape).toEqual([2, 2]);
      expect(Array.from(result.data)).toEqual([2, 3, 5, 6]);
    });

    it("should handle reverse slicing with negative step", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5]), [5]);
      const result = await tensor.slice([null, null, -1]);

      expect(result.shape).toEqual([5]);
      expect(Array.from(result.data)).toEqual([5, 4, 3, 2, 1]);
    });

    it("should preserve requires_grad", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [4], true);
      const result = await tensor.slice([1, 3]);

      expect(result.requires_grad).toBe(true);
    });

    it("should handle mixed slicing with numbers and slices", async () => {
      const tensor = new Tensor(
        new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        [3, 3],
      );
      const result = await tensor.slice(1, ":");

      expect(result.shape).toEqual([3]);
      expect(Array.from(result.data)).toEqual([4, 5, 6]);
    });

    it("should slice the first half of a dimension", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [6]);
      const result = await tensor.slice([0, 3]);

      expect(result.shape).toEqual([3]);
      expect(Array.from(result.data)).toEqual([1, 2, 3]);
    });

    it("should handle overlapping step slices", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5]), [5]);
      const result = await tensor.slice([0, 4, 2]);

      expect(result.shape).toEqual([2]);
      expect(Array.from(result.data)).toEqual([1, 3]);
    });

    it("should throw error for invalid dimensions", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3]), [3]);
      await expect(tensor.slice(":", ":")).rejects.toThrow(
        "Too many indices for tensor",
      );
    });

    it("should handle 3D tensor slicing", async () => {
      const tensor = new Tensor(
        new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        [3, 2, 2],
      );
      const result = await tensor.slice(":", [0, 2], ":");

      expect(result.shape).toEqual([3, 2, 2]);
      expect(Array.from(result.data)).toEqual([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      ]);
    });
  });
  describe("pow", () => {
    it("should correctly square a tensor (power of 2)", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
      const [result] = await tensor.pow(2);

      expect(result.shape).toEqual([2, 2]);
      expect(Array.from(result.data)).toEqual([1, 4, 9, 16]);
    });

    it("should correctly calculate square root (power of 0.5)", async () => {
      const tensor = new Tensor(new Float32Array([1, 4, 9, 16]), [2, 2]);
      const [result] = await tensor.pow(0.5);

      expect(result.shape).toEqual([2, 2]);
      expect(Array.from(result.data)).toEqual([1, 2, 3, 4]);
    });

    it("should handle negative numbers with even powers", async () => {
      const tensor = new Tensor(new Float32Array([-2, -3, 2, 3]), [2, 2]);
      const [result] = await tensor.pow(2);

      expect(result.shape).toEqual([2, 2]);
      expect(Array.from(result.data)).toEqual([4, 9, 4, 9]);
    });

    it("should preserve requires_grad", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2], true);
      const [result] = await tensor.pow(2);

      expect(result.requires_grad).toBe(true);
    });

    it("should handle power of 1 (identity)", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
      const [result] = await tensor.pow(1);

      expect(Array.from(result.data)).toEqual([1, 2, 3, 4]);
    });

    it("should handle power of 0 (all ones)", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
      const [result] = await tensor.pow(0);

      expect(Array.from(result.data)).toEqual([1, 1, 1, 1]);
    });

    it("should maintain shape for 1D tensors", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [4]);
      const [result] = await tensor.pow(2);

      expect(result.shape).toEqual([4]);
      expect(Array.from(result.data)).toEqual([1, 4, 9, 16]);
    });
  });
  describe("sum", () => {
    it("should sum 1D tensor correctly", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [4]);
      const result = await tensor.sum([0]);

      expect(result.shape).toEqual([1]);
      expect(Array.from(result.data)[0]).toBe(10); // 1 + 2 + 3 + 4
    });

    it("should sum 2D tensor along first dimension", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
      const result = await tensor.sum([0]);

      expect(result.shape).toEqual([3]);
      expect(Array.from(result.data)).toEqual([5, 7, 9]); // [1+4, 2+5, 3+6]
    });

    it("should sum 2D tensor along second dimension", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
      const result = await tensor.sum([1]);

      expect(result.shape).toEqual([2]);
      expect(Array.from(result.data)).toEqual([6, 15]); // [1+2+3, 4+5+6]
    });

    it("should preserve requires_grad", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [4], true);
      const result = await tensor.sum([0]);

      expect(result.requires_grad).toBe(true);
    });

    it("should handle tensor with all zeros", async () => {
      const tensor = new Tensor(new Float32Array([0, 0, 0, 0]), [2, 2]);
      const result = await tensor.sum([1]);

      expect(result.shape).toEqual([2]);
      expect(Array.from(result.data)).toEqual([0, 0]);
    });

    it("should handle single element tensor", async () => {
      const tensor = new Tensor(new Float32Array([5]), [1]);
      const result = await tensor.sum([0]);

      expect(result.shape).toEqual([1]);
      expect(Array.from(result.data)[0]).toBe(5);
    });

    it("should handle summing along multiple dimensions", async () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
      const result = await tensor.sum([0, 1]);

      expect(result.shape).toEqual([1]);
      expect(Array.from(result.data)[0]).toBe(21); // sum of all elements
    });

    it("should maintain correct shape after summing", async () => {
      const tensor = new Tensor(
        new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        [3, 3],
      );
      const result = await tensor.sum([0]);

      expect(result.shape).toEqual([3]);
      expect(Array.from(result.data)).toEqual([12, 15, 18]); // column sums
    });
  });

  it("should default requires_grad to false", () => {
    const data = new Float32Array([1, 2, 3, 4]);
    const shape = [2, 2];
    const tensor = new Tensor(data, shape);

    expect(tensor.requires_grad).toBe(false);
  });

  it("should throw an error if the number of elements in data and shape are different", () => {
    const data = new Float32Array([1, 2, 3]);
    const shape = [2, 2];

    expect(() => new Tensor(data, shape)).toThrow("Incompatible shapes");
  });

  describe("norm", () => {
    it("should calculate Euclidean norm along default dimension", async () => {
      const tensor = new Tensor(new Float32Array([3, 4]), [2]);
      const [result] = await tensor.norm();

      // Should be sqrt(3^2 + 4^2) = 5
      expect(result.shape).toEqual([1]);
      expect(Array.from(result.data)[0]).toBeCloseTo(5);
    });

    it("should handle 2D tensor Euclidean norm", async () => {
      const tensor = new Tensor(new Float32Array([3, 4, 6, 8]), [2, 2]);
      const [result] = await tensor.norm();

      console.log("result:", result.data.toString());

      // For dim=0: sqrt(3^2 + 6^2) and sqrt(4^2 + 8^2)
      expect(result.shape).toEqual([2]);
      expect(Array.from(result.data)[0]).toBeCloseTo(6.708203932499369); // sqrt(45)
      expect(Array.from(result.data)[1]).toBeCloseTo(8.94427190999916); // sqrt(80)
    });

    it("should preserve requires_grad", async () => {
      const tensor = new Tensor(new Float32Array([3, 4]), [2], true);
      const [result] = await tensor.norm();

      expect(result.requires_grad).toBe(true);
    });

    it("should handle tensor with all zeros", async () => {
      const tensor = new Tensor(new Float32Array([0, 0, 0, 0]), [2, 2]);
      const [result] = await tensor.norm();

      expect(Array.from(result.data)).toEqual([0, 0]);
    });

    it("should handle 1D tensor with single element", async () => {
      const tensor = new Tensor(new Float32Array([5]), [1]);
      const [result] = await tensor.norm();

      expect(result.shape).toEqual([1]);
      expect(Array.from(result.data)[0]).toBe(5);
    });
  });
});
