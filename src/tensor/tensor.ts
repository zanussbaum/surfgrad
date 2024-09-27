export class Tensor {
  data: Float32Array;
  shape: number[];
  requires_grad: boolean;
  grad: Tensor | null;

  constructor(data: Float32Array, shape: number[], requires_grad = false) {
    this.data = data;
    this.shape = shape;
    this.requires_grad = requires_grad;
    this.grad = null;
  }

  transpose() {
    const shape = this.shape;
    const data = new Float32Array(this.data.length);
    for (let i = 0; i < shape[0]; i++) {
      for (let j = 0; j < shape[1]; j++) {
        data[i * shape[1] + j] = this.data[j * shape[0] + i];
      }
    }
    return new Tensor(data, shape.slice().reverse());
  }
}
