import { Tensor } from "../tensor/tensor.js";
import { UnaryOp } from "../autograd/function.js";

export class ReLU extends UnaryOp {
  protected readonly shaderPath: string = "/src/shaders/relu.wgsl";

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    if (!this.context) {
      throw new Error("Context is null; did you already call Exp.backward?");
    }
    const [input] = this.context.saved_tensors;

    this.context = null;
    // gradient of relu is 1 if x > 0, 0 otherwise

    const gradient = new Float32Array(input.data.length);
    for (let i = 0; i < input.data.length; i++) {
      gradient[i] = input.data[i] > 0 ? grad_output.data[i] : 0;
    }

    return [new Tensor(gradient, input.shape, input.requires_grad)];
  }
}
