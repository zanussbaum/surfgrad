import { Add } from "../ops/add.js";
import { Node } from "../autograd/node.js";

export class Tensor {
  data: Float32Array;
  shape: number[];
  requires_grad: boolean;
  gradFn: Node | null = null;
  grad: Tensor | null = null;

  constructor(
    data: Float32Array,
    shape: number[],
    requires_grad = false,
    gradFn: Node | null = null,
  ) {
    // if number of elements in data and shape are different, throw error
    if (data.length !== shape.reduce((a, b) => a * b)) {
      throw new Error(
        "Incompatible shapes. Data and shape do not match. {data: " +
          data.length +
          ", shape: " +
          shape.reduce((a, b) => a * b) +
          "}",
      );
    }
    this.data = data;
    this.shape = shape;
    this.requires_grad = requires_grad;
    this.gradFn = gradFn;
  }

  static full(shape: number[], value: number, requires_grad = false) {
    const data = new Float32Array(shape.reduce((a, b) => a * b)).fill(value);

    return new Tensor(data, shape, requires_grad);
  }

  static onesLike(tensor: Tensor) {
    return Tensor.full(tensor.shape, 1, tensor.requires_grad);
  }

  static zerosLike(tensor: Tensor) {
    return Tensor.full(tensor.shape, 0, tensor.requires_grad);
  }

  async add(tensor: Tensor) {
    const addOp = await Add.create();

    return addOp.forward(this, tensor);
  }

  transpose() {
    const [rows, cols] = this.shape;
    const transposedData = new Float32Array(this.data.length);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        transposedData[j * rows + i] = this.data[i * cols + j];
      }
    }

    return new Tensor(transposedData, [cols, rows], this.requires_grad);
  }

  async backward() {
    if (!this.requires_grad) {
      throw new Error(
        "backward() can only be called on tensors that require gradients",
      );
    }

    // Start with gradient 1.0 for scalar outputs
    let grad = Tensor.onesLike(this);
    await this.backwardStep(grad);
  }

  private async backwardStep(grad: Tensor) {
    if (!this.gradFn) {
      // This is a leaf tensor
      if (this.requires_grad) {
        if (!this.grad) {
          this.grad = grad;
        } else {
          const [grad] = await this.add(this.grad);
          this.grad = grad;
        }
      }
      return;
    }

    const gradInputs = await this.gradFn.op.backward(grad);

    for (let i = 0; i < this.gradFn.next_functions.length; i++) {
      const [nextFn, inputIdx] = this.gradFn.next_functions[i];
      await nextFn.output.backwardStep(gradInputs[inputIdx]);
    }
  }
}
