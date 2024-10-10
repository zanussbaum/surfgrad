import { Add } from "../ops/add.js";
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
    console.log(
      "setting grad",
      grad.data.toString(),
      "for",
      this.context,
      "data",
      this.data.toString(),
    );
    if (!this.grad) {
      this.grad = grad;
    } else {
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
    let curr_grad = Tensor.onesLike(this);
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
      console.log("node", node.data.toString());
      console.log("node.op", node.context);
      console.log(
        "parents",
        node.context?.parents.map(([t]) => t.data.toString()),
      );
      console.log("\n");
      for (const [parent] of node.context?.parents || []) {
        dfs(parent);
      }
      topo_order.push(node);
    };

    dfs(this);
    // TODO this is wrong!!
    console.log(
      "topo_order",
      topo_order.map((t) => t.context),
    );

    return topo_order;
  }
}
