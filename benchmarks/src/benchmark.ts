import { MatMul, Mul, Tensor, Add, Exp, Log } from "surfgrad";
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

function calculateGFLOPS(
  operation: string,
  size: number,
  timeMs: number,
): number {
  let operations: number;

  switch (operation.toLowerCase()) {
    case "matmul":
      // For matrix multiplication: 2 * n^3 - n^2 operations
      operations = 2 * Math.pow(size, 3) - Math.pow(size, 2);
      break;
    case "mul":
    case "add":
      // Element-wise operations: n^2 for n x n matrices
      operations = Math.pow(size, 2);
      break;
    case "exp":
    case "log":
      // Element-wise operations: n for n elements
      operations = size;
      break;
    default:
      throw new Error(
        "Unsupported operation. Please use 'matmul', 'mul', 'add', 'exp', or 'log'.",
      );
  }

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
  console.log(`Running ${shader} with size ${size}`);
  switch (shader) {
    case "matmul":
      op = await MatMul.create();
      break;
    case "mul":
      op = await Mul.create();
      break;
    case "add":
      op = await Add.create();
      break;
    case "exp":
      op = await Exp.create();
      break;
    case "log":
      op = await Log.create();
      break;
    default:
      throw new Error(`Unknown shader: ${shader}`);
  }

  // Warmup runs
  for (let i = 0; i < warmupRuns; i++) {
    await op.forward(a, b);
  }

  // Benchmark runs
  const times: number[] = [];
  for (let i = 0; i < benchmarkRuns; i++) {
    const [result, total_time] = await op.forward(a, b);
    times.push(total_time);
  }

  const averageTime = times.reduce((a, b) => a + b, 0) / times.length;
  const gflops = calculateGFLOPS(shader, size, averageTime);

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
