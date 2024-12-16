import { Tensor } from "../tensor/tensor.js";
import { Module } from "./module.js";

export class Embedding extends Module {
    vocab_size: number;
    emb_dim: number;
    embedding: Tensor;
    constructor(vocab_size: number, emb_dim: number) {
        super("embedding");

        this.vocab_size = vocab_size;
        this.emb_dim = emb_dim;
        this.embedding = Tensor.randn([vocab_size, emb_dim], true);
    }

    async forward(...inputs: [Tensor]): Promise<[Tensor]> {
        const [embeddings] = await this.embedding.gather(inputs[0]);
        return [embeddings];
    }
}
