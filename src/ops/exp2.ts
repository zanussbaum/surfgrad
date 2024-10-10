import { Tensor } from "../tensor/tensor.js";
import { UnaryOp } from "../autograd/function.js";
import { Mul } from "./mul.js";

export class Exp2 extends UnaryOp {
  protected readonly shaderPath: string = "/src/shaders/exp2.wgsl";

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    const exp_x = this.output;
    if (!exp_x) {
      throw new Error("Exp output is null");
    }
    const x = this.inputs[0];

    const log2 = new Tensor(new Float32Array([Math.log(2)]), [1], false);
    const mulOp = await Mul.create();

    const [exp_x_mul_log_x] = await mulOp.forward(exp_x, log2);

    const [grad] = await mulOp.forward(grad_output, exp_x_mul_log_x);

    await x.setGrad(grad);

    return [grad];
  }
}
