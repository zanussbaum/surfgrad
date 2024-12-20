import { Tensor } from "../tensor/tensor.js";
import { Module } from "./module.js";
import { Linear } from "./linear.js";

export class MultiHeadAttention extends Module {
  qkv: Linear; // Combined projection for Query, Key, Value
  output: Linear; // Output projection
  num_heads: number;
  head_dim: number;
  hidden_dim: number;

  constructor(hidden_dim: number, num_heads: number) {
    super("multihead_attention");

    this.num_heads = num_heads;
    this.head_dim = Math.floor(hidden_dim / num_heads);
    this.hidden_dim = hidden_dim;

    if (this.head_dim * num_heads !== hidden_dim) {
      throw new Error(
        `Hidden dimension ${hidden_dim} must be divisible by number of heads ${num_heads}`,
      );
    }

    // Combined QKV projection
    // Projects to 3x hidden_dim for Q, K, V
    this.qkv = new Linear(hidden_dim, hidden_dim * 3);

    // Output projection
    this.output = new Linear(hidden_dim, hidden_dim);
  }

  private async reshapeToHeads(tensor: Tensor): Promise<Tensor[]> {
    const heads: Tensor[] = [];

    // Each head will be (seqlen, head_dim)
    for (let i = 0; i < this.num_heads; i++) {
      const start = i * this.head_dim;
      const end = start + this.head_dim;
      const head = await tensor.slice(":", [start, end]);
      heads.push(head);
    }

    return heads;
  }

  private async scaledDotProductAttention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
  ): Promise<[Tensor, number]> {
    // Scale factor is 1/sqrt(head_dim)
    const scale = 1 / Math.sqrt(this.head_dim);
    const scaleTensor = Tensor.full(query.shape, scale, false);

    // Compute attention scores
    const [scores] = await query.matmul(key.transpose());
    const [scaledScores] = await scores.mul(scaleTensor);

    // Softmax implementation
    const [expScores] = await scaledScores.exp();
    const sumExp = await expScores.sum([1]);

    const [attention] = await expScores.div(sumExp);

    // Apply attention to values
    return attention.matmul(value);
  }

  async forward(input: Tensor): Promise<[Tensor]> {
    // Project input to Q, K, V
    const [qkv] = await this.qkv.forward(input);

    // Split into Q, K, V
    const query = await qkv.slice(":", [0, this.hidden_dim]);
    const key = await qkv.slice(":", [this.hidden_dim, this.hidden_dim * 2]);
    const value = await qkv.slice(":", [
      this.hidden_dim * 2,
      this.hidden_dim * 3,
    ]);

    // Split each of Q, K, V into heads
    const queryHeads = await this.reshapeToHeads(query);
    const keyHeads = await this.reshapeToHeads(key);
    const valueHeads = await this.reshapeToHeads(value);

    // Compute attention for each head
    // this will be slow, we should create bmm
    const headOutputs: Tensor[] = [];
    for (let i = 0; i < this.num_heads; i++) {
      const [headOutput] = await this.scaledDotProductAttention(
        queryHeads[i],
        keyHeads[i],
        valueHeads[i],
      );
      headOutputs.push(headOutput);
    }

    // Concatenate heads
    let concatOutput = headOutputs[0];
    for (let i = 1; i < headOutputs.length; i++) {
      concatOutput = await concatOutput.concat(headOutputs[i], 1);
    }

    // Final output projection
    return this.output.forward(concatOutput);
  }
}
