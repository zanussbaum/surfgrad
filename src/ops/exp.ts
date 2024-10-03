import { Tensor } from "../tensor/tensor.js";
import { UnaryOp } from "../autograd/function.js";
import { Log } from "./log.js";
import { Mul } from "./mul.js";

export class Exp extends UnaryOp {
  protected readonly shaderPath: string = "/src/shaders/exp.wgsl";

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    if (!this.context) {
      throw new Error("Context is null; did you already call Exp.backward?");
    }
    const [input] = this.context.saved_tensors;

    this.context = null;

    // The gradient of exp(x) is exp(x) * grad_output
    const [exp_x] = await this.forward(input);
    const logOp = await Log.create();
    const [log_x] = await logOp.forward(input);

    const mulOp = await Mul.create();

    const [exp_x_mul_log_x] = await mulOp.forward(exp_x, log_x);

    return [exp_x_mul_log_x];
  }
}
