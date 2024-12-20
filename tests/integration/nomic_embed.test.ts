import { test, expect } from "@playwright/test";

test("NomicEmbed forward pass with known values", async ({ page }) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  // Inject test function
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-expect-error ignore error for tests
      import("/dist/bundle.js").then((module) => {
        const { Tensor, NomicEmbed } = module;

        window.runNomicEmbedTest = async function () {
          // Create configuration matching the HF config
          const config = {
            vocab_size: 30528,
            hidden_size: 768,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 3072,
            hidden_act: "swiglu",
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 8192,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: "rotary",
            use_cache: true,
            classifier_dropout: null,
            rotary_emb_fraction: 1.0,
            qkv_proj_bias: false,
            mlp_fc1_bias: false,
            mlp_fc2_bias: false,
            causal: false,
          };

          // Create sample input tensors
          const seqLength = 1; // Small sequence for testing

          // Create input IDs tensor with some token IDs
          const inputIds = new Tensor(
            new Float32Array([1]),
            [seqLength],
            false,
          );

          // Create attention mask (all 1s for no masking)
          const attentionMask = new Tensor(
            new Float32Array([1]),
            [seqLength],
            false,
          );

          // Create position IDs (optional)
          const positionIds = new Tensor(
            new Float32Array([0]),
            [seqLength],
            false,
          );

          // Create token type IDs (optional)
          const tokenTypeIds = new Tensor(
            new Float32Array([0]),
            [seqLength],
            false,
          );

          // Initialize model
          const model = new NomicEmbed(config);

          // Forward pass
          const [output] = await model.forward(
            inputIds,
            attentionMask,
            positionIds,
            tokenTypeIds,
          );

          return {
            inputShape: inputIds.shape,
            outputShape: output.shape,
            outputData: Array.from(output.data),
          };
        };
        resolve();
      });
    });
  });

  // Run the test function in the browser context
  const result = await page.evaluate(() => window.runNomicEmbedTest());

  // Test input shape
  expect(result.inputShape).toEqual([1]); // [sequence_length]

  // Test output shape - should be [hidden_size] after pooling and normalization
  expect(result.outputShape).toEqual([768]); // [hidden_size]

  // Verify output is normalized (L2 norm should be close to 1)
  const l2Norm = Math.sqrt(
    result.outputData.reduce((sum, val) => sum + val * val, 0),
  );
  expect(l2Norm).toBeCloseTo(1, 6);

  // Verify output values are within reasonable range
  result.outputData.forEach((value) => {
    expect(Math.abs(value)).toBeLessThan(1); // Normalized values should be < 1
  });

  await page.close();
});

declare global {
  interface Window {
    runNomicEmbedTest: () => Promise<{
      inputShape: number[];
      outputShape: number[];
      outputData: number[];
    }>;
  }
}
