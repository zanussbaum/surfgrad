import { Tensor } from "./tensor/tensor.js";
import { Context } from "./autograd/context.js";
import { MatMul } from "./ops/matmul.js";
import { Mul } from "./ops/mul.js";

export { Tensor, Context, MatMul, Mul };

async function main() {
  const x = new Tensor(
    new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    [2, 3],
    true,
  );
  const y = new Tensor(new Float32Array([2.0]), [1], false);
  const ctx = new Context();

  // Forward pass
  const z = await Mul.forward(ctx, x, y);
  console.log(z);

  const loss = new Tensor(new Float32Array(z.data), z.shape, true);

  // Backward pass
  const [grad_x, grad_y] = await Mul.backward(ctx, loss);
  console.log(grad_x, grad_y);
}

main();
