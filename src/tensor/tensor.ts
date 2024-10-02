export class Tensor {
  data: Float32Array;
  shape: number[];
  requires_grad: boolean;
  grad: Tensor | null;

  constructor(data: Float32Array, shape: number[], requires_grad = false) {
    // if number of elements in data and shape are different, throw error
    if (data.length !== shape.reduce((a, b) => a * b)) {
      throw new Error("Incompatible shapes");
    }
    this.data = data;
    this.shape = shape;
    this.requires_grad = requires_grad;
    this.grad = null;
  }

  static full(shape: number[], value: number, requires_grad = false) {
    const data = new Float32Array(shape.reduce((a, b) => a * b)).fill(value);

    return new Tensor(data, shape, requires_grad);
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
}
