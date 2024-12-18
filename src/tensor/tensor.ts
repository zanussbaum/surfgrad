import { Add } from "../ops/add.js";
import { MatMul } from "../ops/matmul.js";
import { Mul } from "../ops/mul.js";
import { Exp } from "../ops/exp.js";
import { Log2 } from "../ops/log2.js";
import { ReLU } from "../ops/relu.js";
import { Exp2 } from "../ops/exp2.js";
import { Ln } from "../ops/ln.js";
import { Div } from "../ops/div.js";

import { AutogradFunction } from "../autograd/function.js";

type SliceArg =
  | number
  | [number | null]
  | [number | null, number | null]
  | [number | null, number | null, number | null]
  | ":";

export class Tensor {
  data: Float32Array;
  shape: number[];
  requires_grad: boolean;
  context: AutogradFunction | null = null;
  grad: Tensor | null = null;

  constructor(
    data: Float32Array,
    shape: number[],
    requires_grad = false,
    context: AutogradFunction | null = null,
  ) {
    // if number of elements in data and shape are different, throw error
    if (data.length !== shape.reduce((a, b) => a * b)) {
      throw new Error(
        "Incompatible shapes. Data and shape do not match. {data: " +
          data.length +
          ", shape: " +
          shape.reduce((a, b) => a * b) +
          "}",
      );
    }
    this.data = data;
    this.shape = shape;
    this.requires_grad = requires_grad;
    this.context = context;
  }

  static full(shape: number[], value: number, requires_grad = false) {
    const data = new Float32Array(shape.reduce((a, b) => a * b)).fill(value);

    return new Tensor(data, shape, requires_grad);
  }

  static onesLike(tensor: Tensor) {
    return Tensor.full(tensor.shape, 1, tensor.requires_grad);
  }

  static zerosLike(tensor: Tensor) {
    return Tensor.full(tensor.shape, 0, tensor.requires_grad);
  }

  static randn(shape: number[], requires_grad = false) {
    const data = new Float32Array(shape.reduce((a, b) => a * b));

    for (let i = 0; i < data.length; i++) {
      data[i] = Math.random() * 2 - 1;
    }

    return new Tensor(data, shape, requires_grad);
  }

  static broadcast(tensor: Tensor, size: number, requires_grad = false) {
    const shape = [size, ...tensor.shape];
    const data = new Float32Array(shape.reduce((a, b) => a * b));

    for (let i = 0; i < data.length; i++) {
      data[i] = tensor.data[i % tensor.shape.reduce((a, b) => a * b)];
    }

    return new Tensor(data, shape, requires_grad);
  }

  async add(tensor: Tensor) {
    const addOp = await Add.create();

    return addOp.forward(this, tensor);
  }

  async mul(tensor: Tensor) {
    const mulOp = await Mul.create();

    return mulOp.forward(this, tensor);
  }

  async sub(tensor: Tensor) {
    if (tensor.shape.length === 1 && this.shape.length === 2) {
      // Broadcasting [n] to [m, n]
      const newShape = [this.shape[0], tensor.shape[0]];
      tensor = Tensor.full(newShape, tensor.data[0], tensor.requires_grad);
    }

    const negOne = Tensor.full(tensor.shape, -1, false);
    const [negTensor] = await tensor.mul(negOne);
    return this.add(negTensor);
  }

  async mean(dims: number[]): Promise<Tensor> {
    // Calculate new shape after reduction
    const shape = this.shape.slice();
    const size = dims.reduce((acc, dim) => acc * shape[dim], 1);

    dims.sort((a, b) => b - a); // Sort in descending order to remove correctly
    dims.forEach((dim) => shape.splice(dim, 1));
    if (shape.length === 0) shape.push(1);

    const result = new Float32Array(shape.reduce((a, b) => a * b, 1));

    // For 1D case
    if (this.shape.length === 1 && dims.includes(0)) {
      let sum = 0;
      for (let i = 0; i < this.data.length; i++) {
        sum += this.data[i];
      }
      result[0] = sum / size;
      return new Tensor(result, shape, this.requires_grad);
    }

    // For higher dimensions (keeping existing logic for 2D)
    const stride = this.shape[1];
    for (let i = 0; i < this.shape[0]; i++) {
      let sum = 0;
      for (let j = 0; j < stride; j++) {
        sum += this.data[i * stride + j];
      }
      result[i] = sum / size;
    }

    return new Tensor(result, shape, this.requires_grad);
  }

  async variance(dims: number[]): Promise<Tensor> {
    const mean = await this.mean(dims);
    const shape = this.shape.slice();
    const size = dims.reduce((acc, dim) => acc * shape[dim], 1);

    dims.sort((a, b) => b - a);
    dims.forEach((dim) => shape.splice(dim, 1));
    if (shape.length === 0) shape.push(1);

    const result = new Float32Array(shape.reduce((a, b) => a * b, 1));

    // For 1D case
    if (this.shape.length === 1 && dims.includes(0)) {
      let sumSquaredDiff = 0;
      const meanValue = mean.data[0];
      for (let i = 0; i < this.data.length; i++) {
        const diff = this.data[i] - meanValue;
        sumSquaredDiff += diff * diff;
      }
      result[0] = sumSquaredDiff / size;
      return new Tensor(result, shape, this.requires_grad);
    }

    // For higher dimensions
    const stride = this.shape[1];
    for (let i = 0; i < this.shape[0]; i++) {
      let sumSquaredDiff = 0;
      const meanValue = mean.data[i];
      for (let j = 0; j < stride; j++) {
        const diff = this.data[i * stride + j] - meanValue;
        sumSquaredDiff += diff * diff;
      }
      result[i] = sumSquaredDiff / size;
    }

    return new Tensor(result, shape, this.requires_grad);
  }

  async sqrt(): Promise<Tensor> {
    const result = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      result[i] = Math.sqrt(this.data[i]);
    }
    return new Tensor(result, this.shape.slice(), this.requires_grad);
  }

  async div(tensor: Tensor): Promise<[Tensor, number]> {
    const divOp = await Div.create();
    return divOp.forward(this, tensor);
  }

  async matmul(tensor: Tensor) {
    const matmulOp = await MatMul.create();

    return matmulOp.forward(this, tensor);
  }

  async exp() {
    const expOp = await Exp.create();

    return expOp.forward(this);
  }

  async log2() {
    const log2Op = await Log2.create();

    return log2Op.forward(this);
  }

  async ln() {
    const lnOp = await Ln.create();

    return lnOp.forward(this);
  }

  async exp2() {
    const exp2Op = await Exp2.create();

    return exp2Op.forward(this);
  }

  async relu() {
    const reluOp = await ReLU.create();

    return reluOp.forward(this);
  }

  async gather(indices: Tensor): Promise<[Tensor, number]> {
    // Convert indices to one-hot
    const oneHot = new Float32Array(indices.shape[0] * this.shape[0]).fill(0);
    for (let i = 0; i < indices.shape[0]; i++) {
      const index = indices.data[i] + i * this.shape[0];
      // set one hot value for the whole vector
      oneHot.fill(1, index, index + 1);
    }

    const oneHotTensor = new Tensor(
      oneHot,
      [indices.shape[0], this.shape[0]],
      indices.requires_grad,
    );

    return oneHotTensor.matmul(this);
  }

  transpose() {
    const [rows, cols] = this.shape;
    const transposedData = new Float32Array(this.data.length);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        transposedData[j * rows + i] = this.data[i * cols + j];
      }
    }

    return new Tensor(transposedData, [cols, rows], this.requires_grad);
  }

  async setGrad(grad: Tensor) {
    if (!this.grad) {
      this.grad = grad;
    } else {
      // for the case there are multiple grads routing to the same tensor/node
      const [grad] = await this.add(this.grad);
      this.grad = grad;
    }
  }

  async backward() {
    if (!this.requires_grad) {
      throw new Error(
        "backward() can only be called on tensors that require gradients",
      );
    }

    // Start with gradient 1.0 for scalar outputs
    const curr_grad = Tensor.onesLike(this);
    if (!this.grad) {
      await this.setGrad(curr_grad);
    }

    // Traverse graph in reverse topological order
    const topo_order = this.build_topo_order().reverse();
    for (const tensorNode of topo_order) {
      const autograd = tensorNode.context;
      if (autograd) {
        await autograd.backward(tensorNode.grad!);
      }
    }
  }

  private build_topo_order(): Tensor[] {
    const topo_order: Tensor[] = [];
    const visited = new Set<Tensor>();

    const dfs = (node: Tensor) => {
      if (visited.has(node)) return;
      visited.add(node);
      for (const parent of node.context?.parents || []) {
        dfs(parent);
      }
      topo_order.push(node);
    };

    dfs(this);

    return topo_order;
  }

  async concat(tensor: Tensor, axis: number): Promise<Tensor> {
    // Validate axis
    if (axis < 0 || axis >= this.shape.length) {
      throw new Error(
        `Invalid axis ${axis}. Must be between 0 and ${this.shape.length - 1}`,
      );
    }

    // For axis 0 concatenation, all other dimensions must match exactly
    if (axis === 0) {
      // For 1D tensors, they must have the same shape
      if (this.shape.length === 1 && this.shape[0] !== tensor.shape[0]) {
        throw new Error(
          `Shape mismatch: tensors have different shapes at non-concatenating dimensions`,
        );
      }
    }

    // For other axes, validate shapes - all dimensions except concat axis must match
    for (let i = 0; i < this.shape.length; i++) {
      if (i !== axis && this.shape[i] !== tensor.shape[i]) {
        throw new Error(
          `Shape mismatch: tensors have different shapes at non-concatenating dimensions`,
        );
      }
    }

    // Calculate new shape
    const newShape = [...this.shape];
    newShape[axis] += tensor.shape[axis];

    // Create new data array
    const newData = new Float32Array(newShape.reduce((a, b) => a * b));

    // Calculate strides for both tensors
    const stride = this.shape[axis];
    const preAxisSize = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const postAxisSize = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);

    // Copy data from both tensors
    for (let i = 0; i < preAxisSize; i++) {
      for (let j = 0; j < postAxisSize; j++) {
        // Copy from first tensor
        for (let k = 0; k < this.shape[axis]; k++) {
          const srcIdx = i * stride * postAxisSize + k * postAxisSize + j;
          const dstIdx =
            i * (stride + tensor.shape[axis]) * postAxisSize +
            k * postAxisSize +
            j;
          newData[dstIdx] = this.data[srcIdx];
        }
        // Copy from second tensor
        for (let k = 0; k < tensor.shape[axis]; k++) {
          const srcIdx =
            i * tensor.shape[axis] * postAxisSize + k * postAxisSize + j;
          const dstIdx =
            i * (stride + tensor.shape[axis]) * postAxisSize +
            (k + stride) * postAxisSize +
            j;
          newData[dstIdx] = tensor.data[srcIdx];
        }
      }
    }

    return new Tensor(
      newData,
      newShape,
      this.requires_grad || tensor.requires_grad,
    );
  }
  async slice(...args: SliceArg[]): Promise<Tensor> {
    if (args.length > this.shape.length) {
      throw new Error(
        `Too many indices for tensor of dimension ${this.shape.length}`,
      );
    }

    // Convert all arguments to normalized slice specs
    const slices = args.map((arg, dim) =>
      this.normalizeSlice(arg, this.shape[dim]),
    );
    console.log("slices:", slices);

    // Calculate output shape and stride info
    const { outputShape, isReducedDim } = this.calculateOutputShape(
      slices,
      this.shape,
    );

    // Handle empty result case
    if (outputShape.length === 0 || outputShape.some((dim) => dim === 0)) {
      return new Tensor(new Float32Array(0), outputShape, this.requires_grad);
    }

    // Create output tensor
    const outputSize = outputShape.reduce((a, b) => a * b, 1);
    const result = new Float32Array(outputSize);

    // For each output position, calculate corresponding input position
    await this.populateSlicedData(
      result,
      outputSize,
      outputShape,
      slices,
      isReducedDim,
    );

    return new Tensor(result, outputShape, this.requires_grad);
  }

  private async populateSlicedData(
    result: Float32Array,
    outputSize: number,
    outputShape: number[],
    slices: [number, number, number][],
    isReducedDim: boolean[],
  ): Promise<void> {
    // Process in chunks to avoid blocking the main thread
    const CHUNK_SIZE = 1000;

    for (let i = 0; i < outputSize; i += CHUNK_SIZE) {
      const end = Math.min(i + CHUNK_SIZE, outputSize);

      for (let j = i; j < end; j++) {
        const outputCoords = this.indexToCoords(j, outputShape);
        const inputCoords = this.mapToInputCoords(
          outputCoords,
          slices,
          isReducedDim,
        );
        const inputIndex = this.coordsToIndex(inputCoords, this.shape);
        result[j] = this.data[inputIndex];
      }

      // Yield to event loop periodically
      if (end < outputSize) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
  }

  private calculateOutputShape(
    slices: [number, number, number][],
    inputShape: number[],
  ) {
    // Pad slices to match input dimensions
    const fullSlices = [...slices];
    while (fullSlices.length < inputShape.length) {
      fullSlices.push([0, inputShape[fullSlices.length], 1]);
    }

    // Track which dimensions are being reduced (single number index)
    const isReducedDim = fullSlices.map(
      ([start, end, step]) => end - start === 1 && step === 1,
    );

    // Calculate output shape, handling both positive and negative steps
    const outputShape = fullSlices
      .map(([start, end, step], i) => {
        if (isReducedDim[i]) return 0;

        if (step > 0) {
          return Math.max(0, Math.ceil((end - start) / step));
        } else {
          // For negative steps, we need to handle the range differently
          // When going backwards, we need to include the start position
          const numElements = Math.max(
            0,
            Math.ceil((start - end + 1) / Math.abs(step)),
          );
          return numElements;
        }
      })
      .filter((size) => size !== 0);

    return { outputShape, isReducedDim };
  }

  private normalizeSlice(
    arg: SliceArg,
    dimSize: number,
  ): [number, number, number] {
    // Handle single number index
    if (typeof arg === "number") {
      const idx = arg < 0 ? dimSize + arg : arg;
      if (idx < 0 || idx >= dimSize) {
        throw new Error(
          `Index ${arg} is out of bounds for dimension ${dimSize}`,
        );
      }
      return [idx, idx + 1, 1];
    }

    // Handle full slice
    if (arg === ":") {
      return [0, dimSize, 1];
    }

    // Handle array spec [start, end, step]
    let [start, end, step] = arg as [
      number | null,
      number | null,
      number | null,
    ];
    step = step ?? 1;

    if (step === 0) {
      throw new Error("Slice step cannot be zero");
    }

    // Handle negative step
    if (step < 0) {
      // Default start is end of dimension for negative step
      start = start ?? dimSize - 1;
      // Default end is before beginning of dimension
      end = end ?? -1;

      // Convert negative indices to positive
      start = start < 0 ? dimSize + start : start;
      // For negative step, don't convert negative end index if it's the default -1
      end = end < 0 && end !== -1 ? dimSize + end : end;

      // Clamp to valid range for negative step
      start = Math.min(dimSize - 1, Math.max(0, start));
      end = Math.min(dimSize - 1, Math.max(0, end));
    } else {
      // Default start is beginning of dimension for positive step
      start = start ?? 0;
      // Default end is end of dimension
      end = end ?? dimSize;

      // Convert negative indices to positive
      start = start < 0 ? dimSize + start : start;
      end = end < 0 ? dimSize + end : end;

      // Clamp to valid range
      start = Math.min(dimSize - 1, Math.max(0, start));
      end = Math.min(dimSize, Math.max(0, end));
    }

    return [start, end, step];
  }

  private indexToCoords(index: number, shape: number[]): number[] {
    const coords = [];
    let remaining = index;
    let stride = shape.reduce((a, b) => a * b, 1);

    for (const dimSize of shape) {
      stride = stride / dimSize;
      const coord = Math.floor(remaining / stride);
      remaining = remaining % stride;
      coords.push(coord);
    }

    return coords;
  }

  private mapToInputCoords(
    outputCoords: number[],
    slices: [number, number, number][],
    isReducedDim: boolean[],
  ): number[] {
    const inputCoords: number[] = [];
    let outputIdx = 0;

    for (let i = 0; i < isReducedDim.length; i++) {
      if (isReducedDim[i]) {
        // For reduced dimensions, use the start index
        inputCoords.push(slices[i][0]);
      } else {
        // For slice dimensions, calculate the actual position
        const [start, , step] = slices[i];
        inputCoords.push(start + outputCoords[outputIdx] * step);
        outputIdx++;
      }
    }

    return inputCoords;
  }

  private coordsToIndex(coords: number[], shape: number[]): number {
    let index = 0;
    let stride = 1;

    for (let i = coords.length - 1; i >= 0; i--) {
      index += coords[i] * stride;
      stride *= shape[i];
    }

    return index;
  }
}
