import { Tensor } from "../tensor/tensor.js";
import { UnaryOp } from "../autograd/function.js";
import { Exp } from "./exp.js";
import { Mul } from "./mul.js";

export class Log extends UnaryOp {
  protected readonly shaderPath: string = "/src/shaders/Log.wgsl";
  async backward(grad_output: Tensor): Promise<Tensor[]> {
    if (!this.context) {
      throw new Error("Context is null; did you already call log.backward?");
    }
    const [input] = this.context.saved_tensors;

    const inverseArray = new Float32Array(input.data.length);

    for (let i = 0; i < input.data.length; i++) {
      inverseArray[i] = 1 / input.data[i];
    }

    const grad = new Tensor(inverseArray, input.shape, input.requires_grad);

    this.context = null;

    return [grad];
  }
}