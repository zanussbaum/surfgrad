import { AutogradFunction } from "../autograd/function.js";
import { Context } from "../autograd/context.js";
import { Tensor } from "../tensor/tensor.js";
import { initWebGPU } from "../webgpu/init.js";

export class MatMul extends AutogradFunction {
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private shaderModule: GPUShaderModule | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  /**
   * Performs matrix multiplication on two tensors.
   *
   * @param ctx The autograd context to save the inputs for the backward pass.
   * @param a The input tensor.
   * @param b The input tensor.
   * @returns The result of the matrix multiplication.
   */
  private constructor() {
    super();
  }

  static async create(): Promise<MatMul> {
    const instance = new MatMul();
    await instance.initialize();
    return instance;
  }

  async initialize() {
    if (this.initialized) return;

    this.device = await initWebGPU();

    const shaderCode = await (await fetch("/src/shaders/matmul.wgsl")).text();
    this.shaderModule = this.device.createShaderModule({ code: shaderCode });

    const visibility = GPUShaderStage.COMPUTE;

    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: visibility, buffer: { type: "uniform" } },
        { binding: 1, visibility: visibility, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: visibility, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: visibility, buffer: { type: "storage" } },
      ]
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: this.shaderModule, entryPoint: "main" },
    });

    this.initialized = true;
  }

  async forward(ctx: Context | null, a: Tensor, b: Tensor) {
    if (!this.device || !this.pipeline || !this.bindGroupLayout) {
      throw new Error("MatMul is not properly initialized");
    }

    if (a.shape[1] !== b.shape[0]) {
      throw new Error(`Incompatible shapes: ${a.shape} and ${b.shape}`);
    }


    const uniformBuffer = this.device.createBuffer({
      size: 3 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([a.shape[0], a.shape[1], b.shape[1]]),
    );

    const bufferA = this.device.createBuffer({
      size: a.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bufferB = this.device.createBuffer({
      size: b.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const resultBuffer = this.device.createBuffer({
      size: a.shape[0] * b.shape[1] * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.device.queue.writeBuffer(bufferA, 0, a.data);
    this.device.queue.writeBuffer(bufferB, 0, b.data);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufferA } },
        { binding: 2, resource: { buffer: bufferB } },
        { binding: 3, resource: { buffer: resultBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);

    const WORKGROUP_SIZE = 256;
    pass.dispatchWorkgroups(Math.ceil(a.shape[0] / WORKGROUP_SIZE));
    pass.end();

    const stagingBuffer = this.device.createBuffer({
      size: a.shape[0] * b.shape[1] * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    encoder.copyBufferToBuffer(
      resultBuffer,
      0,
      stagingBuffer,
      0,
      stagingBuffer.size,
    );

    this.device.queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultArray = new Float32Array(stagingBuffer.getMappedRange());
    const resultCopy = new Float32Array(resultArray);
    stagingBuffer.unmap();

    const resultTensor = new Tensor(resultCopy, [a.shape[0], b.shape[1]], true);

    if (ctx) {
      ctx.save_for_backward(a, b);
    }

    // clean up buffers
    bufferA.destroy();
    bufferB.destroy();
    uniformBuffer.destroy();
    resultBuffer.destroy();

    return resultTensor;
  }

  async backward(ctx: Context, grad_output: Tensor) {
    const [a, b] = ctx.saved_tensors;

    const b_t = b.transpose();

    const grad_a = await this.forward(null, grad_output, b_t);

    const a_t = a.transpose();

    // grad_b calculation: a.T @ grad_output
    const grad_b = await this.forward(null, a_t, grad_output);

    return [grad_a, grad_b];
  }

  cleanup(): void {
    super.cleanup();
    // We don't need to explicitly destroy the shader module or pipeline
    // They will be garbage collected when no longer referenced
    this.device = null;
    this.pipeline = null;
    this.shaderModule = null;
    this.initialized = false;
  }
}
