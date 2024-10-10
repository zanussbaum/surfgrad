// import { AutogradContext } from "./context.js";
import { Tensor } from "../tensor/tensor.js";
import { initWebGPU } from "../webgpu/init.js";
import { timeAndExecute } from "../util/time.js";

export abstract class AutogradFunction {
  protected initialized: boolean = false;
  protected context: Node | null = null;
  protected device: GPUDevice | null = null;
  protected pipeline: GPUComputePipeline | null = null;
  protected shaderModule: GPUShaderModule | null = null;
  protected bindGroupLayout: GPUBindGroupLayout | null = null;
  protected abstract readonly shaderPath: string;
  public parents: Tensor[] = [];
  protected inputs: Tensor[] = [];
  protected output: Tensor | null = null;
  protected requiresGrad: boolean[] = [];

  constructor() {}

  async initialize(): Promise<void> {
    if (this.initialized) return;
    // Perform any common initialization here
    this.initialized = true;
  }

  static async create<T extends AutogradFunction>(
    this: new () => T,
  ): Promise<T> {
    const instance = new this();
    await instance.initialize();
    return instance;
  }

  async setAutogradContext(inputs: Tensor[], output: Tensor): Promise<void> {
    // parents is a list of tensors that have requires_grad = true
    const parents: Tensor[] = [];
    for (let i = 0; i < inputs.length; i++) {
      if (inputs[i].requires_grad) {
        parents.push(inputs[i]);
      }
    }

    this.parents = parents;
    this.requiresGrad = inputs.map((input) => input.requires_grad);

    this.inputs = inputs;
    this.output = output;

    // set requires_grad to false so values aren't saved when forward is called in backward()
    for (let i = 0; i < inputs.length; i++) {
      inputs[i].requires_grad = false;
    }
    output.context = this;
  }

  abstract forward(...inputs: Tensor[]): Promise<[Tensor, number]>;

  abstract backward(grad_output: Tensor): Promise<Tensor[]>;

  cleanup(): void {
    // Perform any common cleanup here
    this.initialized = false;
  }
}

export abstract class BinaryOp extends AutogradFunction {
  async initialize() {
    if (this.initialized) return;

    this.device = await initWebGPU();

    const shaderCode = await (await fetch(this.shaderPath)).text();
    this.shaderModule = this.device.createShaderModule({ code: shaderCode });

    const visibility = GPUShaderStage.COMPUTE;

    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: visibility, buffer: { type: "uniform" } },
        {
          binding: 1,
          visibility: visibility,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: visibility,
          buffer: { type: "read-only-storage" },
        },
        { binding: 3, visibility: visibility, buffer: { type: "storage" } },
      ],
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

  abstract validateShapes(a: Tensor, b: Tensor): Tensor;

  async forward(a: Tensor, b: Tensor): Promise<[Tensor, number]> {
    if (!this.device || !this.pipeline || !this.bindGroupLayout) {
      throw new Error(`${this.constructor.name} is not properly initialized`);
    }

    b = this.validateShapes(a, b);

    const shapes = new Uint32Array([a.shape[0], a.shape[1], b.shape[1]]);
    const uniformBuffer = this.device.createBuffer({
      size: 3 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, shapes);

    const bufferA = this.device.createBuffer({
      size: a.data.length * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "bufferA",
    });
    this.device.queue.writeBuffer(bufferA, 0, a.data);

    const bufferB = this.device.createBuffer({
      size: b.data.length * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "bufferB",
    });
    this.device.queue.writeBuffer(bufferB, 0, b.data);

    const resultBuffer = this.device.createBuffer({
      size: a.shape[0] * b.shape[1] * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: "resultBuffer",
    });

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
    pass.dispatchWorkgroups(Math.ceil((a.shape[0] * b.shape[1]) / WORKGROUP_SIZE));
    pass.end();

    const stagingBuffer = this.device.createBuffer({
      size: a.shape[0] * b.shape[1] * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: "stagingBuffer",
    });

    encoder.copyBufferToBuffer(
      resultBuffer,
      0,
      stagingBuffer,
      0,
      stagingBuffer.size,
    );

    const [resultArray, executionTime] = await timeAndExecute(
      this.device,
      encoder,
      stagingBuffer,
    );

    const resultCopy = new Float32Array(resultArray);
    stagingBuffer.unmap();

    const resultTensor = new Tensor(resultCopy, [a.shape[0], b.shape[1]], true);

    // clean up buffers
    bufferA.destroy();
    bufferB.destroy();
    uniformBuffer.destroy();
    if (a.requires_grad || b.requires_grad) {
      this.setAutogradContext([a, b], resultTensor);
    }
    return [resultTensor, executionTime];
  }
}

export abstract class UnaryOp extends AutogradFunction {
  async initialize() {
    if (this.initialized) return;

    this.device = await initWebGPU();

    const shaderCode = await (await fetch(this.shaderPath)).text();
    this.shaderModule = this.device.createShaderModule({ code: shaderCode });

    const visibility = GPUShaderStage.COMPUTE;

    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: visibility, buffer: { type: "uniform" } },
        {
          binding: 1,
          visibility: visibility,
          buffer: { type: "read-only-storage" },
        },
        { binding: 2, visibility: visibility, buffer: { type: "storage" } },
      ],
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

  async forward(a: Tensor): Promise<[Tensor, number]> {
    if (!this.device || !this.pipeline || !this.bindGroupLayout) {
      throw new Error(`${this.constructor.name} is not properly initialized`);
    }

    const uniformBuffer = this.device.createBuffer({
      size: 2 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "uniformBuffer",
    });

    this.device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([a.shape[0], a.shape[1]]),
    );

    const bufferA = this.device.createBuffer({
      size: a.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "bufferA",
    });

    const resultBuffer = this.device.createBuffer({
      size: a.shape[0] * a.shape[1] * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: "resultBuffer",
    });

    this.device.queue.writeBuffer(bufferA, 0, a.data);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufferA } },
        { binding: 2, resource: { buffer: resultBuffer } },
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
      label: "stagingBuffer",
    });

    encoder.copyBufferToBuffer(
      resultBuffer,
      0,
      stagingBuffer,
      0,
      stagingBuffer.size,
    );

    const [resultArray, executionTime] = await timeAndExecute(
      this.device,
      encoder,
      stagingBuffer,
    );

    const resultCopy = new Float32Array(resultArray);
    stagingBuffer.unmap();

    const resultTensor = new Tensor(resultCopy, [a.shape[0], a.shape[1]], true);
    if (a.requires_grad) {
      await this.setAutogradContext([a], resultTensor);
    }

    // clean up buffers
    bufferA.destroy();
    uniformBuffer.destroy();
    resultBuffer.destroy();
    return [resultTensor, executionTime];
  }

  abstract backward(grad_output: Tensor): Promise<Tensor[]>;

  cleanup(): void {
    super.cleanup();
    this.device = null;
    this.pipeline = null;
    this.shaderModule = null;
    this.initialized = false;
  }
}
