import { Tensor } from "../tensor/tensor.js";
import { UnaryOp } from "../autograd/function.js";
import { Mul } from "./mul.js";

export class Log extends UnaryOp {
  protected readonly shaderPath: string = "/src/shaders/Log.wgsl";

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    if (!this.context) {
      throw new Error("Context is null; did you already call log.backward?");
    }
    const [input] = this.context.inputs;

    const inverseArray = new Float32Array(input.data.length);

    for (let i = 0; i < input.data.length; i++) {
      inverseArray[i] = 1 / (input.data[i] * Math.log(2));
    }

    const mulOp = await Mul.create();
    const [grad] = await mulOp.forward(
      new Tensor(inverseArray, input.shape, input.requires_grad),
      grad_output,
    );

    this.context = null;

    return [grad];
  }
}
