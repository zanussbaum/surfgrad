import { Tensor } from "../tensor/tensor.js";
import { Module } from "./module.js";

export class LayerNorm extends Module {
    normalized_shape: number[];
    eps: Tensor;
    gamma: Tensor;
    beta: Tensor;
    constructor(normalized_shape: number[], eps: number) {
        super("layer_norm");
        this.normalized_shape = normalized_shape;
        // eps should be [2, 1] for broadcasting
        this.eps = Tensor.full([1], eps);  // Make eps a scalar tensor
        // gamma and beta should match the feature dimension
        this.gamma = Tensor.full([1, normalized_shape[0]], 1);  // [1, 3] for broadcasting
        this.beta = Tensor.full([1, normalized_shape[0]], 0);   // [1, 3] for broadcasting
    }

    async forward(x: Tensor): Promise<[Tensor]> {
        const reduction_dims = [1];  // Reduce over the feature dimension

        // Calculate mean and reshape for broadcasting
        const mean = await x.mean(reduction_dims);
        mean.shape = [mean.shape[0], 1];  // [2, 1]
        
        const variance = await x.variance(reduction_dims);
        variance.shape = [variance.shape[0], 1];  // [2, 1]
        
        console.log("x shape:", x.shape);            // [2, 3]
        console.log("mean shape:", mean.shape);      // [2, 1]
        console.log("variance shape:", variance.shape); // [2, 1]
        console.log("gamma shape:", this.gamma.shape);  // [1, 3]
        console.log("beta shape:", this.beta.shape);    // [1, 3]

        const [numerator] = await x.sub(mean);  // [2, 3]
        const [denominator] = await variance.add(this.eps);
        const sqrtDenom = await denominator.sqrt();
        const [normalized] = await numerator.div(sqrtDenom);
        
        const [gamma] = await normalized.mul(this.gamma);  // [2, 3] * [1, 3] -> [2, 3]
        const [beta] = await gamma.add(this.beta);        // [2, 3] + [1, 3] -> [2, 3]
        return [beta];
    }
}