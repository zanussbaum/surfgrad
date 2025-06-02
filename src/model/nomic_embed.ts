import { Tensor } from "../tensor/tensor.js";
import { Module } from "../layers/module.js";
import { LayerNorm } from "../layers/norm.js";
import { MultiHeadAttention } from "../layers/attention.js";
import { MLP } from "../layers/mlp.js";
import { Embedding } from "../layers/embedding.js";

export interface NomicEmbedConfig {
  vocab_size: number;
  hidden_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  intermediate_size: number;
  hidden_act: string;
  hidden_dropout_prob: number;
  attention_probs_dropout_prob: number;
  max_position_embeddings: number;
  type_vocab_size: number;
  initializer_range: number;
  layer_norm_eps: number;
  pad_token_id: number;
  position_embedding_type: string;
  use_cache: boolean;
  classifier_dropout: number | null;
  rotary_emb_fraction: number;
  use_flash_attn: boolean;
  qkv_proj_bias: boolean;
  mlp_fc1_bias: boolean;
  mlp_fc2_bias: boolean;
  causal: boolean;
}

class NomicBertEmbeddings extends Module {
  private wordEmbeddings: Embedding;
  private positionEmbeddings: Embedding | null;
  private typeEmbeddings: Embedding | null;
  private maxPositionEmbeddings: number;
  private typeVocabSize: number;

  constructor(config: NomicEmbedConfig) {
    super("bert_embeddings");

    // Word embeddings
    this.wordEmbeddings = new Embedding(config.vocab_size, config.hidden_size);

    // Position embeddings if using absolute positions
    this.maxPositionEmbeddings = config.max_position_embeddings;
    this.positionEmbeddings =
      this.maxPositionEmbeddings > 0 && config.rotary_emb_fraction <= 0
        ? new Embedding(config.max_position_embeddings, config.hidden_size)
        : null;

    // Token type embeddings if used
    this.typeVocabSize = config.type_vocab_size;
    this.typeEmbeddings =
      this.typeVocabSize > 0
        ? new Embedding(config.type_vocab_size, config.hidden_size)
        : null;
  }

  async forward(
    inputIds: Tensor,
    positionIds?: Tensor,
    tokenTypeIds?: Tensor,
    inputsEmbeds?: Tensor,
  ): Promise<[Tensor]> {
    // Get word embeddings
    let [embeddings] = inputsEmbeds
      ? [inputsEmbeds]
      : await this.wordEmbeddings.forward(inputIds);

    // Add token type embeddings if used
    // if (this.typeEmbeddings && this.typeVocabSize > 0 && tokenTypeIds) {
    //   const [typeEmbeddings] = await this.typeEmbeddings.forward(tokenTypeIds);
    //   console.log("typeEmbeddings.data", typeEmbeddings.data.toString());
    //   console.log("typeEmbeddings.shape", typeEmbeddings.shape);
    //   [embeddings] = await embeddings.add(typeEmbeddings);
    // }

    return [embeddings];
  }
}

class NomicBertLayer extends Module {
  private attention: MultiHeadAttention;
  private mlp: MLP;
  private layerNorm1: LayerNorm;
  private layerNorm2: LayerNorm;

  constructor(config: NomicEmbedConfig) {
    super("bert_layer");
    this.attention = new MultiHeadAttention(
      config.hidden_size,
      config.num_attention_heads,
    );
    this.mlp = new MLP(config.hidden_size, config.intermediate_size);
    this.layerNorm1 = new LayerNorm(
      [config.hidden_size],
      config.layer_norm_eps,
    );
    this.layerNorm2 = new LayerNorm(
      [config.hidden_size],
      config.layer_norm_eps,
    );
  }

  async forward(...inputs: [Tensor]): Promise<[Tensor]> {
    // Self-attention
    const [hiddenStates] = inputs;
    const [normed1] = await this.layerNorm1.forward(hiddenStates);
    const [attnOutput] = await this.attention.forward(normed1);
    const [residual1] = await hiddenStates.add(attnOutput);

    // MLP
    const [normed2] = await this.layerNorm2.forward(residual1);
    const [mlpOutput] = await this.mlp.forward(normed2);
    const [residual2] = await residual1.add(mlpOutput);
    return [residual2];
  }
}

class NomicBertEncoder extends Module {
  private layers: NomicBertLayer[];

  constructor(config: NomicEmbedConfig) {
    super("bert_encoder");
    this.layers = Array(config.num_hidden_layers)
      .fill(null)
      .map(() => new NomicBertLayer(config));
  }

  async forward(...args: Tensor[]): Promise<[Tensor]> {
    let [hiddenStates, attentionMask] = args;
    let currentOutput = hiddenStates;

    // Pass through each layer
    for (const layer of this.layers) {
      [currentOutput] = await layer.forward(currentOutput);
    }

    return [currentOutput];
  }
}

export class NomicEmbed extends Module {
  private embeddings: NomicBertEmbeddings;
  private encoder: NomicBertEncoder;
  private emb_ln: LayerNorm;

  constructor(config: NomicEmbedConfig) {
    super("nomic_embed");

    // Initialize components
    this.embeddings = new NomicBertEmbeddings(config);
    this.encoder = new NomicBertEncoder(config);
    this.emb_ln = new LayerNorm([config.hidden_size], config.layer_norm_eps);
  }

  private async meanPooling(
    modelOutput: Tensor,
    attentionMask: Tensor,
  ): Promise<[Tensor]> {
    return [await modelOutput.mean([0])];
  }

  async forward(...args: Tensor[]): Promise<[Tensor]> {
    // Get embeddings
    const [inputIds, attentionMask, positionIds, tokenTypeIds] = args;
    const [hidden] = await this.embeddings.forward(
      inputIds,
      positionIds,
      tokenTypeIds,
    );
    console.log("hidden.data", hidden.data.toString());

    // Apply layer norm
    const [normed] = await this.emb_ln.forward(hidden);
    console.log("normed.data", normed.data.toString());

    // Pass through encoder
    const [encoded] = await this.encoder.forward(normed, attentionMask);
    // Mean pooling
    console.log("encoded.data", encoded.data.toString());
    const [pooled] = await this.meanPooling(encoded, attentionMask);
    console.log("pooled.shape", pooled.shape);

    const [norm] = await pooled.norm(2, 0);
    console.log("norm.shape", norm.shape);
    console.log("norm", norm.data.toString());

    const [pooledNormed] = await pooled.div(norm);
    // Normalize embeddings
    return [pooledNormed];
  }
}
