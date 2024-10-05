import { BinaryOp } from "../autograd/function.js";
import { Tensor } from "../tensor/tensor.js";

export class MatMul extends BinaryOp {
  protected readonly shaderPath: string = "/src/shaders/matmul.wgsl";

  validateShapes(a: Tensor, b: Tensor): [Tensor, Tensor] {
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(
        `Incompatible shapes for MatMul: ${a.shape} and ${b.shape}`,
      );
    }

    return [a, b];
  }

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    if (!this.context) {
      throw new Error("Context is null; did you already call MatMul.backward?");
    }
    const [a, b] = this.context.inputs;

    this.context = null;

    const b_t = b.transpose();
    const [grad_a] = await this.forward(grad_output, b_t);

    const a_t = a.transpose();
    const [grad_b] = await this.forward(a_t, grad_output);

    return [grad_a, grad_b];
  }
}
