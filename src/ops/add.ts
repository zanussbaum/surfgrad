import { BinaryOp } from "../autograd/function.js";
import { Tensor } from "../tensor/tensor.js";

export class Add extends BinaryOp {
  protected readonly shaderPath: string = "/src/shaders/add.wgsl";

  validateShapes(a: Tensor, b: Tensor): [Tensor, Tensor] {
    if (!a.shape.every((value, index) => value === b.shape[index])) {
      if (b.shape.length === 1 && b.shape[0] === 1) {
        // Broadcast scalar
        b = Tensor.full(a.shape, b.data[0], b.requires_grad);
      } else {
        throw new Error(
          `Incompatible shapes for Add: ${a.shape} and ${b.shape}`,
        );
      }
    }
    return [a, b];
  }

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    if (!this.context) {
      throw new Error("Context is null; did you already call Add.backward?");
    }
    const [a, b] = this.context.saved_tensors;

    this.context = null;

    const grad_a = a.requires_grad
      ? new Tensor(new Float32Array(grad_output.data).fill(1), a.shape, true)
      : null;
    const grad_b = b.requires_grad
      ? new Tensor(new Float32Array(grad_output.data).fill(1), b.shape, true)
      : null;

    return [grad_a, grad_b].filter(
      (tensor): tensor is Tensor => tensor !== null,
    );
  }
}
