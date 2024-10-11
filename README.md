# surfgrad

================

**surfgrad** is a high-performance, WebGPU-powered AutoGrad library that enables browser-based tensor operations with GPU acceleration. 

Key Features:

- 🚀 Blazing-fast tensor operations leveraging WebGPU
- 🧠 Automatic differentiation for deep learning in the browser
- 🌐 Zero backend dependencies - runs entirely client-side
- 📦 Lightweight and easy to integrate into existing web projects

Perfect for running tensor operations and (in the future) machine learning models in the browser!

It's heavily inspired by [micrograd](https://github.com/karpathy/micrograd),
[tinygrad](https://github.com/tinygrad/tinygrad), and [PyTorch](https://github.com/pytorch/pytorch) and aims to leverage the power of WebGPU/WGSL for in-browser machine learning.

## Usage

---

`surfgrad` supports basic tensor operations such as `matmul`, `mul`, `add`, `exp`, and `log`.

To use `surfgrad`,

```typescript
import { Tensor } from "surfgrad";

const tensorA = new Tensor(new Float32Array([1, 2, 3, 4]), shape: [2, 2], requires_grad: true);
const tensorB = new Tensor(new Float32Array([5, 6, 7, 8]), shape: [2, 2], requires_grad: true);

const [result, executionTime] = await tensorA.matmul(tensorB);

console.log(result);

await result.backward();

```

## Testing

---

`surfgrad` has unit tests and integration tests. To run the unit tests, run the following command:

```bash
npm run unit
```

and to run the integration tests, run the following command:

```bash
npm run integration
```

## Benchmarks

---

We also have benchmarks that can be helpful to demonstrate the performance of the `matmul` kernels.
To run the benchmarks, run the following command:

```bash
npm run benchmark
```

and open a browser to `localhost:9000`.

This will run the benchmarks for the library and display the results.

## Contributing

---

Contributions to `surfgrad` are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

## License

---

SurfGrad is licensed under the MIT License. See the LICENSE file for details.
