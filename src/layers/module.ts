import { Tensor } from "../tensor/tensor";
export abstract class Module {
  protected name: string;
  constructor(name: string) {
    if (name === null || name === undefined) {
      throw Error("Name cannot be null or undefined");
    }

    this.name = name;
  }

  /**
   * Abstract method that must be implemented by all layer subclasses
   * Defines the forward pass computation of the layer
   * @param inputs - Input tensor(s) to the layer
   * @returns Output tensor(s) from the layer
   */
  abstract forward(...args: Tensor[]): Promise<[Tensor]>;
}
