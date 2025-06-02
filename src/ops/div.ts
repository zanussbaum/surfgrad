import { BinaryOp } from "../autograd/function.js";
import { Tensor } from "../tensor/tensor.js";
import { divShader } from "../shaders/div.js";

export class Div extends BinaryOp {
  protected readonly shader: string = divShader;

  validateShapes(a: Tensor, b: Tensor): Tensor {
    // Handle scalar case first
    if (b.shape.length === 1 && b.shape[0] === 1) {
      return Tensor.full(a.shape, b.data[0], b.requires_grad);
    }

    // Get dimensions of both tensors
    const dimA = a.shape.length;
    const dimB = b.shape.length;

    // Calculate the number of dimensions in the output
    const maxDim = Math.max(dimA, dimB);

    // Pad shapes with 1s from the left to match max dimensions
    const paddedA = Array(maxDim - dimA)
      .fill(1)
      .concat(a.shape);
    const paddedB = Array(maxDim - dimB)
      .fill(1)
      .concat(b.shape);

    // Check if shapes can be broadcast
    const outputShape = [];
    for (let i = 0; i < maxDim; i++) {
      if (paddedA[i] === paddedB[i]) {
        outputShape.push(paddedA[i]);
      } else if (paddedA[i] === 1) {
        outputShape.push(paddedB[i]);
      } else if (paddedB[i] === 1) {
        outputShape.push(paddedA[i]);
      } else {
        throw new Error(
          `Incompatible shapes for broadcasting: ${a.shape} and ${b.shape}`,
        );
      }
    }

    // If shapes are already compatible, return original tensor
    if (outputShape.every((dim, i) => dim === b.shape[i])) {
      return b;
    }

    // Create new broadcasted tensor
    const newSize = outputShape.reduce((acc, dim) => acc * dim, 1);
    const newData = new Float32Array(newSize);

    // For a tensor of shape [n] being broadcast to [n, m],
    // we want to repeat each element m times consecutively
    if (b.shape.length === 1 && outputShape.length === 2) {
      const n = b.shape[0];
      const m = outputShape[1];

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
          newData[i * m + j] = b.data[i];
        }
      }
    } else {
      // General case for broadcasting across multiple dimensions
      for (let i = 0; i < newSize; i++) {
        // Convert flat index to coordinates
        let remaining = i;
        const coords = [];
        for (const dim of outputShape) {
          coords.push(remaining % dim);
          remaining = Math.floor(remaining / dim);
        }
        coords.reverse();

        // Map to input tensor coordinates
        let inputIdx = 0;
        let stride = 1;
        for (let dim = b.shape.length - 1; dim >= 0; dim--) {
          const outputDim = dim + (outputShape.length - b.shape.length);
          const coord = coords[outputDim] % b.shape[dim];
          inputIdx += coord * stride;
          stride *= b.shape[dim];
        }

        newData[i] = b.data[inputIdx];
      }
    }

    return new Tensor(newData, outputShape, b.requires_grad);
  }

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    const [a, b] = this.inputs;
    const [aRequiresGrad, bRequiresGrad] = this.requiresGrad;

    const grad_a_result = await this.forward(grad_output, b);
    const grad_a = aRequiresGrad ? grad_a_result[0] : null;
    if (grad_a !== null) {
      await a.setGrad(grad_a);
    }

    const grad_b_result = await this.forward(a, grad_output);
    const grad_b = bRequiresGrad ? grad_b_result[0] : null;
    if (grad_b !== null) {
      await b.setGrad(grad_b);
    }

    return [grad_a, grad_b].filter(
      (tensor): tensor is Tensor => tensor !== null,
    );
  }
}
