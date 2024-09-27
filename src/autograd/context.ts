import { Tensor } from "../tensor/tensor.js";

export class Context {
  saved_tensors: Tensor[] = [];

  save_for_backward(...tensors: Tensor[]) {
    this.saved_tensors = tensors;
  }
}
