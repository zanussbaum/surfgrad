import { test, expect } from "@playwright/test";

test("Embedding forward pass with known values", async ({ page }) => {
  await page.goto("http://localhost:8080");

  page.on("console", (msg) => {
    console.log(msg);
  });

  // Inject test function
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // @ts-expect-error ignore error for tests
      import("/dist/bundle.js").then((module) => {
        const { Tensor, Embedding } = module;
        
        window.runEmbeddingTest = async function () {
          const vocabSize = 128;
          const embeddingDim = 2; // Using small dim for easy verification
          
          // Create embedding layer
          const embedding = new Embedding(vocabSize, embeddingDim);
          console.log(embedding);

          // Create input tensor with indices
          const inputIndices = new Tensor(
            new Float32Array([1, 5, 10]), // Sample indices
            [3], // Sequence length of 3
            false
          );

          // Forward pass
          const [embeddings] = await embedding.forward(inputIndices);

          return {
            inputIndices: Array.from(inputIndices.data),
            embedding: embedding.embedding,
            outputShape: embeddings.shape,
            outputData: Array.from(embeddings.data)
          };
        };
        resolve();
      });
    });
  });

  // Run the test function in the browser context
  const result = await page.evaluate(() => window.runEmbeddingTest());

  // Validate shapes
  expect(result.outputShape).toEqual([3, 2]);  // Sequence length x Embedding dim

  const expectedOutput = [
    result.embedding.data[2],
    result.embedding.data[3],
    result.embedding.data[10],
    result.embedding.data[11] ,
    result.embedding.data[20],
    result.embedding.data[21]
  ];
  
  expect(result.outputData).toEqual(expectedOutput);

  await page.close();
});

declare global {
  interface Window {
    runEmbeddingTest: () => Promise<{
      inputIndices: number[];
      embedding: { data: number[] };
      outputShape: number[];
      outputData: number[];
    }>;
  }
}