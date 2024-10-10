import { Tensor } from "../tensor/tensor.js";
import { UnaryOp } from "../autograd/function.js";

export class Ln extends UnaryOp {
  protected readonly shaderPath: string = "/src/shaders/ln.wgsl";
  async backward(grad_output: Tensor): Promise<Tensor[]> {
    const [input] = this.inputs;
    const inputRequiresGrad = this.requiresGrad;

    // The gradient of ln(x) is 1 / x
    const inverseArray = new Float32Array(input.data.length);
    for (let i = 0; i < input.data.length; i++) {
      inverseArray[i] = grad_output.data[i] / input.data[i];
    }

    const grad_inverse = new Tensor(inverseArray, input.shape, false);

    await input.setGrad(grad_inverse);

    return [grad_inverse];
  }
}
