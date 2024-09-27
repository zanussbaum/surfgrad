import { Tensor } from './tensor/tensor.js';
import { Context } from './autograd/context.js';
import { MatMul } from './ops/matmul.js';

async function main() {
  // Create tensors
  const x = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2], true);
  const w = new Tensor(new Float32Array([0.1, 0.2, 0.3, 0.4]), [2, 2], true);
  const ctx = new Context();

  // Forward pass
  const y = await MatMul.forward(ctx, x, w);

  console.log("Input:", x);
  console.log("Weight:", w);
  console.log("Output:", y);
  console.log("Context:", ctx);

  // Backward pass (assuming some loss function)
  const grad_output = new Tensor(new Float32Array([1, 1, 1, 1]), [2, 2]);
  const [grad_x, grad_w] = await MatMul.backward(ctx, grad_output);

  console.log("Output:", y);
  console.log("Gradient of x:", grad_x);
  console.log("Gradient of w:", grad_w);
}

main().catch(console.error);