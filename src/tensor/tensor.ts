import { Add } from "../ops/add.js";
import { MatMul } from "../ops/matmul.js";
import { Mul } from "../ops/mul.js";
import { Exp } from "../ops/exp.js";
import { Log2 } from "../ops/log2.js";
import { ReLU } from "../ops/relu.js";
import { Exp2 } from "../ops/exp2.js";
import { Ln } from "../ops/ln.js";

import { AutogradFunction } from "../autograd/function.js";

export class Tensor {
  data: Float32Array;
  shape: number[];
  requires_grad: boolean;
  context: AutogradFunction | null = null;
  grad: Tensor | null = null;

  constructor(
    data: Float32Array,
    shape: number[],
    requires_grad = false,
    context: AutogradFunction | null = null,
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
    this.context = context;
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

  async mul(tensor: Tensor) {
    const mulOp = await Mul.create();

    return mulOp.forward(this, tensor);
  }

  async matmul(tensor: Tensor) {
    const matmulOp = await MatMul.create();

    return matmulOp.forward(this, tensor);
  }

  async exp() {
    const expOp = await Exp.create();

    return expOp.forward(this);
  }

  async log2() {
    const log2Op = await Log2.create();

    return log2Op.forward(this);
  }

  async ln() {
    const lnOp = await Ln.create();

    return lnOp.forward(this);
  }

  async exp2() {
    const exp2Op = await Exp2.create();

    return exp2Op.forward(this);
  }

  async relu() {
    const reluOp = await ReLU.create();

    return reluOp.forward(this);
  }

  // In Tensor class
  async gather(indices: Tensor): Promise<[Tensor, number]> {
    // Convert indices to one-hot
    const oneHot = new Float32Array(indices.shape[0] * this.shape[0]).fill(0);
    for (let i = 0; i < indices.shape[0]; i++) {
        oneHot[i * this.shape[0] + indices.data[i]] = 1;
    }
    const oneHotTensor = new Tensor(oneHot, [indices.shape[0], this.shape[0]], indices.requires_grad);
    
    // Use existing matmul
    return oneHotTensor.matmul(this);
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

  async setGrad(grad: Tensor) {
    if (!this.grad) {
      this.grad = grad;
    } else {
      // for the case there are multiple grads routing to the same tensor/node
      const [grad] = await this.add(this.grad);
      this.grad = grad;
    }
  }

  async backward() {
    if (!this.requires_grad) {
      throw new Error(
        "backward() can only be called on tensors that require gradients",
      );
    }

    // Start with gradient 1.0 for scalar outputs
    const curr_grad = Tensor.onesLike(this);
    if (!this.grad) {
      await this.setGrad(curr_grad);
    }

    // Traverse graph in reverse topological order
    const topo_order = this.build_topo_order().reverse();
    for (const tensorNode of topo_order) {
      const autograd = tensorNode.context;
      if (autograd) {
        await autograd.backward(tensorNode.grad!);
      }
    }
  }

  private build_topo_order(): Tensor[] {
    const topo_order: Tensor[] = [];
    const visited = new Set<Tensor>();

    const dfs = (node: Tensor) => {
      if (visited.has(node)) return;
      visited.add(node);
      for (const parent of node.context?.parents || []) {
        dfs(parent);
      }
      topo_order.push(node);
    };

    dfs(this);

    return topo_order;
  }
}
