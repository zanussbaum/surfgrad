import { Tensor } from "../tensor/tensor.js";
import { UnaryOp } from "../autograd/function.js";

export class ReLU extends UnaryOp {
  protected readonly shaderPath: string = "/src/shaders/relu.wgsl";

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    const [input] = this.inputs;

    // gradient of relu is 1 if x > 0, 0 otherwise

    const gradientArr = new Float32Array(input.data.length);
    for (let i = 0; i < input.data.length; i++) {
      gradientArr[i] = input.data[i] > 0 ? grad_output.data[i] : 0;
    }

    const gradient = new Tensor(gradientArr, input.shape, false);
    await input.setGrad(gradient);

    return [gradient];
  }
}
