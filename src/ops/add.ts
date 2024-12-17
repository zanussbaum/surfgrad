import { BinaryOp } from "../autograd/function.js";
import { Tensor } from "../tensor/tensor.js";
import { addShader } from "../shaders/add.js";

export class Add extends BinaryOp {
  protected readonly shader: string = addShader;

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
        for (let i = 0; i < b.shape[0]; i++) {
          for (let j = 0; j < a.shape[1]; j++) {
            newData[i * a.shape[1] + j] = b.data[i];
          }
        }
        b = new Tensor(newData, newShape, b.requires_grad);
      }
      else if (b.shape.length === 2 && b.shape[0] === 1 && b.shape[1] === a.shape[1]) {
        // Broadcast [1, m] to [n, m]
        const newShape = [a.shape[0], b.shape[1]];
        console.log("Broadcasting [1,m] to shape:", newShape);
        const newData = new Float32Array(newShape[0] * newShape[1]);
        for (let i = 0; i < a.shape[0]; i++) {
          for (let j = 0; j < b.shape[1]; j++) {
            newData[i * b.shape[1] + j] = b.data[j];
          }
        }
        b = new Tensor(newData, newShape, b.requires_grad);
      }
      else {
        throw new Error(
          `Incompatible shapes for Add: ${a.shape} and ${b.shape}`,
        );
      }
    }
    return b;
  }

  async backward(grad_output: Tensor): Promise<Tensor[]> {
    const [a, b] = this.inputs;

    const grad_a = this.requiresGrad[0]
      ? new Tensor(new Float32Array(grad_output.data), a.shape, false)
      : null;

    if (grad_a !== null) {
      await a.setGrad(grad_a);
    }

    const grad_b = this.requiresGrad[1]
      ? new Tensor(new Float32Array(grad_output.data), b.shape, false)
      : null;

    if (grad_b !== null) {
      await b.setGrad(grad_b);
    }

    return [grad_a, grad_b].filter(
      (tensor): tensor is Tensor => tensor !== null,
    );
  }
}
