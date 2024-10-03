import { Tensor } from "../tensor/tensor.js";
import { UnaryOp } from "../autograd/function.js";
import { Exp } from "./exp.js";
import { Mul } from "./mul.js";

export class Log extends UnaryOp {
  protected readonly shaderPath: string = "/src/shaders/log.wgsl";
  async backward(grad_output: Tensor): Promise<Tensor[]> {
    if (!this.context) {
      throw new Error("Context is null; did you already call log.backward?");
    }
    const [input] = this.context.saved_tensors;

    this.context = null;

    // The gradient of exp(x) is exp(x) * grad_output
    const [log2_x] = await this.forward(input);

    return [grad_output];
  }
}
