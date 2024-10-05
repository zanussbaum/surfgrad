import { AutogradFunction } from "./function.js";
import { Tensor } from "../tensor/tensor.js";

export class Node {
  op: AutogradFunction;
  inputs: Tensor[];
  output: Tensor;
  next_functions: [Node, number][]; // [Node, output_idx]

  constructor(op: AutogradFunction, inputs: Tensor[], output: Tensor) {
    this.op = op;
    this.inputs = inputs;
    this.output = output;
    this.next_functions = [];
  }
}
