import { Tensor } from "../tensor/tensor.js";
import { Module } from "./module.js";
import { Linear } from "./linear.js";

type ActivationType = "relu" | "silu" | "gelu" | "swiglu" | "none";

export class MLP extends Module {
  up: Linear; // Project up to larger dimension
  down: Linear; // Project back down
  activation: ActivationType;

  constructor(
    dim: number, // input/output dimension
    hiddenDim: number, // hidden dimension
    activation: ActivationType = "relu",
  ) {
    super("mlp");

    // For SwiGLU, we need double the hidden dimension for gating
    const actualHiddenDim = activation === "swiglu" ? hiddenDim * 2 : hiddenDim;

    this.up = new Linear(dim, actualHiddenDim);
    this.down = new Linear(hiddenDim, dim);
    this.activation = activation;
  }

  private async gelu(x: Tensor): Promise<[Tensor, number]> {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const sqrt2OverPi = Math.sqrt(2 / Math.PI);

    // Calculate x^3
    const [xSquared] = await x.mul(x);
    const [xCubed] = await xSquared.mul(x);

    // Calculate 0.044715 * x^3
    const [scaledCube] = await xCubed.mul(
      Tensor.full(x.shape, 0.044715, false),
    );

    // Add x to the scaled cube
    const [innerSum] = await x.add(scaledCube);

    // Multiply by sqrt(2/π)
    const [scaled] = await innerSum.mul(
      Tensor.full(x.shape, sqrt2OverPi, false),
    );

    // Calculate tanh using (e^x - e^-x)/(e^x + e^-x)
    const [exp] = await scaled.exp();
    const [negScaled] = await scaled.mul(Tensor.full(x.shape, -1, false));
    const [negExp] = await negScaled.exp();

    const [numerator] = await exp.sub(negExp);
    const [denominator] = await exp.add(negExp);

    const [tanh] = await numerator.div(denominator);

    // Add 1 to tanh result
    const [tanhPlusOne] = await tanh.add(Tensor.full(x.shape, 1, false));

    // Multiply by x
    const [xTimesSum] = await x.mul(tanhPlusOne);

    // Multiply by 0.5 for final result
    return xTimesSum.mul(Tensor.full(x.shape, 0.5, false));
  }

  private async silu(x: Tensor): Promise<[Tensor, number]> {
    const [negX] = await x.mul(Tensor.full(x.shape, -1, false));
    const [expNegX] = await negX.exp();
    const [onePlusExpNegX] = await expNegX.add(Tensor.full(x.shape, 1, false));

    const [sigmoid] = await Tensor.full(x.shape, 1, false).div(onePlusExpNegX);
    return x.mul(sigmoid);
  }

  private async applyActivation(x: Tensor): Promise<[Tensor, number]> {
    switch (this.activation) {
      case "relu":
        return x.relu();
      case "silu":
        return this.silu(x);
      case "gelu":
        return this.gelu(x);
      case "swiglu": {
        // Split the tensor in half for gate and value paths
        const halfSize = Math.floor(x.shape[x.shape.length - 1] / 2);
        const [gate, value] = await Promise.all([
          x.slice(":", [0, halfSize]),
          x.slice(":", [halfSize, x.shape[x.shape.length - 1]]),
        ]);
        const [gateActivated] = await this.silu(gate);
        return gateActivated.mul(value);
      }
      case "none":
        return [x, -1];
      default:
        throw new Error(`Unknown activation type: ${this.activation}`);
    }
  }

  async forward(...inputs: [Tensor]): Promise<[Tensor]> {
    const [input] = inputs;

    // Project up to hidden dimension
    const [upProjected] = await this.up.forward(input);

    // Apply activation
    const [activated] = await this.applyActivation(upProjected);

    // Project back down
    return this.down.forward(activated);
  }

  // Helper method for creating standard configurations
  static create(config: {
    dim: number; // input/output dimension
    hiddenMul?: number; // multiplier for hidden dimension (default 4)
    activation?: ActivationType;
  }): MLP {
    const {
      dim,
      hiddenMul = 4, // typical transformer uses 4x dimension for FFN
      activation = "relu",
    } = config;

    return new MLP(dim, dim * hiddenMul, activation);
  }
}
