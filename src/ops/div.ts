import { BinaryOp } from "../autograd/function.js";
import { Tensor } from "../tensor/tensor.js";
import { divShader } from "../shaders/div.js";

export class Div extends BinaryOp {
  protected readonly shader: string = divShader;

  validateShapes(a: Tensor, b: Tensor): Tensor {
    if (!a.shape.every((value, index) => value === b.shape[index])) {
      if (b.shape.length === 1 && b.shape[0] === 1) {
        // Broadcast scalar
        b = Tensor.full(a.shape, b.data[0], b.requires_grad);
      } 
      else if (b.shape.length === 1 && b.shape[0] === a.shape[1]) {
        // Broadcast [m] to [n, m]
        b = Tensor.broadcast(b, a.shape[0], b.requires_grad);
      }
      else if (b.shape.length === 2 && b.shape[1] === 1) {
        // Broadcast [n, 1] to [n, m]
        const newShape = [b.shape[0], a.shape[1]];
        console.log("Broadcasting [n,1] to shape:", newShape);
        const newData = new Float32Array(newShape[0] * newShape[1]);
        // Repeat the values across the second dimension
        for (let i = 0; i < b.shape[0]; i++) {
          for (let j = 0; j < a.shape[1]; j++) {
            newData[i * a.shape[1] + j] = b.data[i];
          }
        }
        b = new Tensor(newData, newShape, b.requires_grad);
      }
      else {
        throw new Error(
          `Incompatible shapes for Div: ${a.shape} and ${b.shape}`,
        );
      }
    }
    return b;
}

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    const [a, b] = this.inputs;
    const [aRequiresGrad, bRequiresGrad] = this.requiresGrad;

    const grad_a_result = await this.forward(grad_output, b);
    const grad_a = aRequiresGrad ? grad_a_result[0] : null;
    if (grad_a !== null) {
      await a.setGrad(grad_a);
    }

    const grad_b_result = await this.forward(a, grad_output);
    const grad_b = bRequiresGrad ? grad_b_result[0] : null;
    if (grad_b !== null) {
      await b.setGrad(grad_b);
    }

    return [grad_a, grad_b].filter(
      (tensor): tensor is Tensor => tensor !== null,
    );
  }
} 