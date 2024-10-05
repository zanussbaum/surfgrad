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
});
