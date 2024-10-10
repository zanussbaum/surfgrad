import { Tensor } from "../tensor/tensor.js";
import { UnaryOp } from "../autograd/function.js";
import { Mul } from "./mul.js";

export class Exp extends UnaryOp {
  protected readonly shaderPath: string = "/src/shaders/exp.wgsl";

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    // derivative of exp(x) is exp(x)
    const exp_x = this.output;
    if (!exp_x) {
      throw new Error("Exp output is null");
    }
    const x = this.inputs[0];
    const mulOp = await Mul.create();
    const [grad] = await mulOp.forward(grad_output, exp_x);

    await x.setGrad(grad);

    return [grad];
  }
}
