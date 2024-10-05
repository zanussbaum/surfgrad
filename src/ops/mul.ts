import { BinaryOp } from "../autograd/function.js";
import { Tensor } from "../tensor/tensor.js";

export class Mul extends BinaryOp {
  protected readonly shaderPath: string = "/src/shaders/mul.wgsl";

  validateShapes(a: Tensor, b: Tensor): [Tensor, Tensor] {
    if (!a.shape.every((value, index) => value === b.shape[index])) {
      if (b.shape.length === 1 && b.shape[0] === 1) {
        // Broadcast scalar
        b = Tensor.full(a.shape, b.data[0], b.requires_grad);
      } else {
        throw new Error(
          `Incompatible shapes for Mul: ${a.shape} and ${b.shape}`,
        );
      }
    }
    return [a, b];
  }

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    if (!this.context) {
      throw new Error("Context is null; did you already call Mul.backward?");
    }
    const [a, b] = this.context.inputs;

    this.context = null;

    const grad_a_result = await this.forward(grad_output, b);
    const grad_a = a.requires_grad ? grad_a_result[0] : null;

    const grad_b_result = await this.forward(a, grad_output);
    const grad_b = b.requires_grad ? grad_b_result[0] : null;

    return [grad_a, grad_b].filter(
      (tensor): tensor is Tensor => tensor !== null,
    );
  }
}
