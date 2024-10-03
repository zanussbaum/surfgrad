import { AutogradFunction } from "../autograd/function.js";
import { Context } from "../autograd/context.js";
import { Tensor } from "../tensor/tensor.js";
import { initWebGPU } from "../webgpu/init.js";

export class Mul extends AutogradFunction {
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private shaderModule: GPUShaderModule | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;

  /**
   * Performs element-wise multiplication on two tensors.
   */
  private constructor() {
    super();
  }

  static async create(): Promise<Mul> {
    const instance = new Mul();
    await instance.initialize();
    return instance;
  }

  async initialize() {
    if (this.initialized) return;

    this.device = await initWebGPU();

    const shaderCode = await (await fetch("/src/shaders/mul.wgsl")).text();
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

  async forward(ctx: Context | null, a: Tensor, scalar: Tensor) {
    if (!this.device || !this.pipeline || !this.bindGroupLayout) {
      throw new Error("Mul is not properly initialized");
    }

    // Check shapes and broadcast scalar if necessary
    if (a.shape.every((value, index) => value !== scalar.shape[index])) {
      if (scalar.shape.length === 1 && scalar.shape[0] === 1) {
        scalar = Tensor.full(a.shape, scalar.data[0], scalar.requires_grad);
      } else {
        throw new Error(`Incompatible shapes: ${a.shape} and ${scalar.shape}`);
      }
    }

    // Create uniform buffer for dimensions
    const uniformBuffer = this.device.createBuffer({
      size: 2 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([a.shape[0], a.shape[1]]),
    );

    const bufferA = this.device.createBuffer({
      size: a.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const scalarStorageBuffer = this.device.createBuffer({
      size: scalar.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const resultBuffer = this.device.createBuffer({
      size: a.shape[0] * a.shape[1] * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.device.queue.writeBuffer(bufferA, 0, a.data);
    this.device.queue.writeBuffer(scalarStorageBuffer, 0, scalar.data);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufferA } },
        { binding: 2, resource: { buffer: scalarStorageBuffer } },
        { binding: 3, resource: { buffer: resultBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);

    const WORKGROUP_SIZE = 64;
    pass.dispatchWorkgroups(Math.ceil(a.shape[0] / WORKGROUP_SIZE));
    pass.end();

    const stagingBuffer = this.device.createBuffer({
      size: a.shape[0] * a.shape[1] * Float32Array.BYTES_PER_ELEMENT,
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

    const resultTensor = new Tensor(resultCopy, [a.shape[0], a.shape[1]], true);

    //Cleanup temporary buffers
    uniformBuffer.destroy();
    bufferA.destroy();
    scalarStorageBuffer.destroy();
    resultBuffer.destroy();
    stagingBuffer.destroy();


    if (ctx) {
      ctx.save_for_backward(a, scalar);
    }

    return resultTensor;
  }

  async backward(ctx: Context, grad_output: Tensor): Promise<Tensor[]> {
    const [a, scalar] = ctx.saved_tensors as [Tensor, Tensor];

    const grad_a = a.requires_grad
      ? await this.forward(null, grad_output, scalar)
      : null;
    const grad_scalar = scalar.requires_grad
      ? await this.forward(null, grad_output, a)
      : null;

    return [grad_a, grad_scalar].filter(
      (tensor): tensor is Tensor => tensor !== null,
    );
  }

  cleanup(): void {
    super.cleanup();
    this.device = null;
    this.pipeline = null;
    this.shaderModule = null;
    this.initialized = false;
  }
}
