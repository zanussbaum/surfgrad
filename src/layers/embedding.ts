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
    this.embedding = Tensor.normal([vocab_size, emb_dim], true, 0.02);
  }

  async forward(...inputs: [Tensor]): Promise<[Tensor]> {
    const [embeddings] = await this.embedding.gather(inputs[0]);
    return [embeddings];
  }
}

export class RotaryEmbedding extends Module {
  base: number;
  dimension: number;
  theta: Tensor;
  sequenceLength: number;
  idxTheta: Tensor | null = null;
  constructor(base: number, dimension: number) {
    super("rope_embedding");
    this.base = base;
    this.dimension = dimension;

    const theta = this.createTheta(dimension, base);
    this.theta = new Tensor(theta, [1, dimension / 2], true);

    this.sequenceLength = 0;
    this.idxTheta = null;
  }

  createTheta(dimension: number, base: number = 10000): Float32Array {
    // Create a new Float32Array of the specified size
    const result = new Float32Array(dimension / 2);

    // Calculate values for each position
    for (let i = 0; i < dimension; i += 2) {
      const value = 1.0 / Math.pow(base, i / dimension);
      result[i / 2] = value;
    }

    return result;
  }

  async buildCache(sequenceLength: number) {
    const posIdx = new Float32Array(sequenceLength);
    for (let i = 0; i < sequenceLength; i++) {
      posIdx[i] = i;
    }

    const posTensor = new Tensor(posIdx, [sequenceLength, 1], true);
    let [idxTheta] = await posTensor.matmul(this.theta);

    idxTheta = await idxTheta.concat(idxTheta, 1);

    return [idxTheta];
  }

  async forward(...inputs: [Tensor]): Promise<[Tensor]> {
    const [x] = inputs;

    const currSeqLen = x.shape[0];
    const d2 = Math.floor(this.dimension / 2);

    if (currSeqLen > this.sequenceLength || this.idxTheta === null) {
      const [cache] = await this.buildCache(currSeqLen);
      this.sequenceLength = currSeqLen;
      this.idxTheta = cache;
    }

    const idxTheta = this.idxTheta;

    const idxThetaLength = idxTheta.data.length;
    const cosIdxThetaArr = new Float32Array(idxThetaLength);
    const sinIdxThetaArr = new Float32Array(idxThetaLength);

    for (let i = 0; i < idxThetaLength; i++) {
      cosIdxThetaArr[i] = Math.cos(idxTheta.data[i]);
      sinIdxThetaArr[i] = Math.sin(idxTheta.data[i]);
    }

    const cosIdxTheta = new Tensor(
      cosIdxThetaArr,
      [currSeqLen, this.dimension],
      x.requires_grad,
    );
    const sinIdxTheta = new Tensor(
      sinIdxThetaArr,
      [currSeqLen, this.dimension],
      x.requires_grad,
    );

    // Rewrite using tensor operations and select
    const leftHalf = await x.slice(":", [null, d2]);
    const rightHalf = await x.slice(":", [d2, this.dimension]);
    const [negHalf] = await rightHalf.mul(Tensor.full([1], -1));

    const half = await negHalf.concat(leftHalf, 1);
    const xRope = await x.slice(":", [null, this.dimension]);

    const [xRopePos] = await xRope.mul(cosIdxTheta);
    const [xRopeNeg] = await half.mul(sinIdxTheta);

    let [rope] = await xRopePos.add(xRopeNeg);
    if (this.dimension < x.shape[1]) {
      const xPass = await x.slice(":", [null, null, d2]);

      rope = await rope.concat(xPass, 1);
    }

    return [rope];
  }
}
