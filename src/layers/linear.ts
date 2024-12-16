import { Tensor } from "../tensor/tensor.js";
import { Module } from "./module.js";

export class Linear extends Module {
    weight: Tensor;
    bias: Tensor;

    constructor(inputSize: number, outputSize: number) {
        super("linear");
        this.weight = Tensor.randn([inputSize, outputSize], true);
        this.bias = Tensor.randn([outputSize], true);
    }

    async forward(...inputs: [Tensor]): Promise<[Tensor]> {
        const [input] = inputs;
        const [output] = await input.matmul(this.weight);
        const [outputBias] = await output.add(this.bias);
        return [outputBias];
    }
}