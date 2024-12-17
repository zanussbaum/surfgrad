import { BinaryOp } from "../autograd/function.js";
import { Tensor } from "../tensor/tensor.js";
import { mulShader } from "../shaders/mul.js";

export class Mul extends BinaryOp {
  protected readonly shader: string = mulShader;

  validateShapes(a: Tensor, b: Tensor): Tensor {
    // Handle broadcasting for 2D tensors
    if (a.shape.length === 2 && b.shape.length === 1) {
      // Broadcasting b [n] to [m, n]
      const newShape = [a.shape[0], b.shape[0]];
      b = Tensor.full(newShape, b.data[0], b.requires_grad);
    } else if (a.shape.length === 2 && b.shape.length === 2) {
      // Handle [m, 1] broadcasting to [m, n]
      if (b.shape[1] === 1) {
        const newShape = [b.shape[0], a.shape[1]];
        b = Tensor.full(newShape, b.data[0], b.requires_grad);
      }
    } else if (b.shape.length === 1 && b.shape[0] === 1) {
      // Scalar broadcasting
      b = Tensor.full(a.shape, b.data[0], b.requires_grad);
    } else if (!a.shape.every((value, index) => value === b.shape[index])) {
      throw new Error(
        `Incompatible shapes for Mul: ${a.shape} and ${b.shape}`,
      );
    }

    // Ensure 2D shapes for WebGPU operations
    if (a.shape.length === 1) {
      a.shape = [a.shape[0], 1];
    }
    if (b.shape.length === 1) {
      b.shape = [b.shape[0], 1];
    }

    return b;
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
