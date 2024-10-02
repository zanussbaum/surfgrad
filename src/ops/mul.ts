import { AutogradFunction } from "../autograd/function.js";
import { Context } from "../autograd/context.js";
import { Tensor } from "../tensor/tensor.js";
import { initWebGPU } from "../webgpu/init.js";

export class Mul extends AutogradFunction {
  /**
   * Performs element-wise multiplication on two tensors.
   *
   * @param ctx The autograd context to save the inputs for the backward pass.
   * @param a The input tensor.
   * @param scalar The scalar to multiply the input with.
   * @returns The result of the element-wise multiplication.
   */
  static async forward(ctx: Context | null, a: Tensor, scalar: Tensor) {
    if (!scalar.shape.every((value, index) => value === [1][index])) {
      throw new Error(`Incompatible shapes: ${a.shape} and ${scalar.shape}`);
    }

    const device = await initWebGPU();

    const shaderCode = await (await fetch("/src/shaders/mul.wgsl")).text();
    const module = device.createShaderModule({ code: shaderCode });

    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });

    // Create uniform buffer for dimensions
    const uniformBuffer = device.createBuffer({
      size: 2 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([a.shape[0], a.shape[1]]),
    );

    const bufferA = device.createBuffer({
      size: a.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const scalarStorageBuffer = device.createBuffer({
      size: scalar.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const resultBuffer = device.createBuffer({
      size: a.shape[0] * a.shape[1] * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    device.queue.writeBuffer(bufferA, 0, a.data);

    // Convert the scalar to a Float32Array before writing
    device.queue.writeBuffer(scalarStorageBuffer, 0, scalar.data);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufferA } },
        { binding: 2, resource: { buffer: scalarStorageBuffer } },
        { binding: 3, resource: { buffer: resultBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);

    // Adjust workgroup size for better occupancy
    const WORKGROUP_SIZE = 32;
    pass.dispatchWorkgroups(
      Math.ceil(a.shape[0] / WORKGROUP_SIZE),
      Math.ceil(a.shape[1] / WORKGROUP_SIZE),
      1,
    );
    pass.end();

    // Create a staging buffer to read the results
    const stagingBuffer = device.createBuffer({
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

    device.queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultArray = new Float32Array(stagingBuffer.getMappedRange());
    const resultCopy = new Float32Array(resultArray);
    stagingBuffer.unmap();

    const resultTensor = new Tensor(resultCopy, [a.shape[0], a.shape[1]], true);

    if (ctx) {
      ctx.save_for_backward(a, scalar);
    }

    return resultTensor;
  }

  static async backward(ctx: Context, grad_output: Tensor) {
    const [a, scalar] = ctx.saved_tensors;

    let grad_a = null;
    if (a.requires_grad == true) {
      grad_a = await Mul.forward(null, grad_output, scalar);
    }

    let grad_scalar = null;
    if (scalar.requires_grad == true) {
      grad_scalar = await Mul.forward(null, grad_output, a);
    }

    return [grad_a, grad_scalar];
  }
}
