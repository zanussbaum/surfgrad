import { Context } from "./context.js";

export abstract class AutogradFunction {
  static forward(ctx: Context, ...inputs: any[]): any {}

  static backward(ctx: Context, grad_output: any): any {
    throw new Error("Not implemented");
  }
}
