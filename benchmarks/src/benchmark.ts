import { MatMul, Mul, Tensor } from "surfgrad";
import { AutogradFunction } from "../../dist/autograd/function";

interface BenchmarkResult {
  shader: string;
  size: number;
  averageTime: number;
  gflops: number;
}

function createRandomMatrix(size: number): Float32Array {
  return Float32Array.from({ length: size * size }, () => Math.random());
}

function calculateGFLOPS(size: number, timeMs: number): number {
  const operations = 2 * Math.pow(size, 3);
  return operations / (timeMs / 1000) / 1e9;
}

async function runSingleBenchmark(
  shader: string,
  size: number,
  warmupRuns: number = 10,
  benchmarkRuns: number = 20,
): Promise<BenchmarkResult> {
  const a = new Tensor(createRandomMatrix(size), [size, size]);
  const b = new Tensor(createRandomMatrix(size), [size, size]);

  // Create op
  let op: AutogradFunction;
  switch (shader) {
    case "matmul":
      op = await MatMul.create();
      break;
    case "mul":
      op = await Mul.create();
      break;
    default:
      throw new Error(`Unknown shader: ${shader}`);
  }

  // Warmup runs
  for (let i = 0; i < warmupRuns; i++) {
    await op.forward(null, a, b);
  }

  // Benchmark runs
  const times: number[] = [];
  for (let i = 0; i < benchmarkRuns; i++) {
    const startTime = performance.now();
    await op.forward(null, a, b);
    const endTime = performance.now();
    times.push(endTime - startTime);
  }

  const averageTime = times.reduce((a, b) => a + b, 0) / times.length;
  const gflops = calculateGFLOPS(size, averageTime);

  return { shader, size, averageTime, gflops };
}

export async function runBenchmark(
  shader: string,
  sizes: number[],
  progressCallback: (progress: number) => void,
): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];

  for (let i = 0; i < sizes.length; i++) {
    const size = sizes[i];
    const result = await runSingleBenchmark(shader, size);
    results.push(result);

    // Report progress
    progressCallback((i + 1) / sizes.length);
  }

  return results;
}
