import { Tensor } from "./tensor/tensor.js";
import { Context } from "./autograd/context.js";
import { MatMul } from "./ops/matmul.js";

export { Tensor, Context, MatMul };

async function main() {
  // Create tensors
  const x = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2], true);
  const w = new Tensor(new Float32Array([0.1, 0.2, 0.3, 0.4]), [2, 2], true);
  const ctx = new Context();

  // Forward pass
  const y = await MatMul.forward(ctx, x, w);

  const loss = new Tensor(
    new Float32Array(y.data.length).fill(1),
    y.shape,
    true,
  );

  console.log("Input:", x);
  console.log("Weight:", w);
  console.log("Output:", y);
  console.log("Context:", ctx);
  console.log("Loss:", loss);

  // Assume some loss gradient flowing back is all ones
  // Backward pass
  const [grad_x, grad_w] = await MatMul.backward(ctx, loss);

  console.log("Gradient of x:", grad_x);
  console.log("Gradient of w:", grad_w);
}

main().catch(console.error);
