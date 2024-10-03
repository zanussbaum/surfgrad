import { Context } from "./context.js";
import { Tensor } from "../tensor/tensor.js";

export abstract class AutogradFunction {
  protected initialized: boolean = false;
  protected context: Context | null = new Context();

  async initialize(): Promise<void> {
    if (this.initialized) return;
    // Perform any common initialization here
    this.initialized = true;
  }

  abstract forward(...inputs: Tensor[]): Promise<Tensor>;

  abstract backward(grad_output: Tensor): Promise<Tensor[]>;

  cleanup(): void {
    // Perform any common cleanup here
    this.initialized = false;
  }
}
