import { Tensor } from "./tensor/tensor.js";
import { Context } from "./autograd/context.js";
import { MatMul } from "./ops/matmul.js";
import { Mul } from "./ops/mul.js"

export { Tensor, Context, MatMul, Mul };

async function main() {
  // Create tensors
  const x = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2], true);
  const scalar = new Tensor(new Float32Array([.1]), [1,], false);
  const ctx = new Context();

  // Forward pass
  const y = await Mul.forward(ctx, x, scalar);

  const loss = new Tensor(
    new Float32Array(y.data),
    y.shape,
    true,
  );

  console.log("Input:", x);
  console.log("Scalar:", scalar);
  console.log("Output:", y)

  const [grad_x,] = await Mul.backward(ctx, loss);

  console.log("Gradient of x:", grad_x);

}

main().catch(console.error);
