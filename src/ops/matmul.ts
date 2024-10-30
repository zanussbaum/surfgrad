import { BinaryOp } from "../autograd/function.js";
import { Tensor } from "../tensor/tensor.js";
import { matmulShader } from "../shaders/matmul.js";

export class MatMul extends BinaryOp {
  protected readonly shader: string = matmulShader;

  validateShapes(a: Tensor, b: Tensor): Tensor {
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(
        `Incompatible shapes for MatMul: ${a.shape} and ${b.shape}`,
      );
    }

    return b;
  }

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    const [a, b] = this.inputs;

    const b_t = b.transpose();
    const [grad_a] = await this.forward(grad_output, b_t);

    await a.setGrad(grad_a);

    const a_t = a.transpose();
    const [grad_b] = await this.forward(a_t, grad_output);

    await b.setGrad(grad_b);

    return [grad_a, grad_b];
  }
}
