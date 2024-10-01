import { Tensor } from "../tensor/tensor.js";

export class Context {
  saved_tensors: Tensor[] = [];

  save_for_backward(...tensors: Tensor[]) {
    // if saved_tensors is empty
    if (this.saved_tensors.length === 0) {
      this.saved_tensors = tensors;
    } else {
      // if saved_tensors is not empty, append tensors
      this.saved_tensors = this.saved_tensors.concat(tensors);
    }
  }
}
