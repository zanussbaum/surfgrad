import { Tensor } from "../tensor/tensor.js";
import { UnaryOp } from "../autograd/function.js";
import { Mul } from "./mul.js";
import { log2Shader } from "../shaders/log2.js";

export class Log2 extends UnaryOp {
  protected readonly shader: string = log2Shader;

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    const [input] = this.inputs;

    const inverseArray = new Float32Array(input.data.length);

    for (let i = 0; i < input.data.length; i++) {
      inverseArray[i] = 1 / (input.data[i] * Math.log(2));
    }

    const mulOp = await Mul.create();
    const [grad] = await mulOp.forward(
      new Tensor(inverseArray, input.shape, false),
      grad_output,
    );

    await input.setGrad(grad);

    return [grad];
  }
}
