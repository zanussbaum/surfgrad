import { Tensor } from "../tensor/tensor.js";
import { UnaryOp } from "../autograd/function.js";
import { Mul } from "./mul.js";

export class Exp extends UnaryOp {
  protected readonly shaderPath: string = "/src/shaders/exp.wgsl";

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    if (!this.context) {
      throw new Error("Context is null; did you already call Exp.backward?");
    }

    const exp_x = this.context.output;

    const log2 = new Tensor(new Float32Array([Math.log(2)]), [1], false);
    const mulOp = await Mul.create();

    const [exp_x_mul_log_x] = await mulOp.forward(exp_x, log2);

    const [grad] = await mulOp.forward(grad_output, exp_x_mul_log_x);

    return [grad];
  }
}
