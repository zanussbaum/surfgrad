import { AutogradFunction } from '../autograd/function.js';
import { Context } from '../autograd/context.js';
import { Tensor } from '../tensor/tensor.js';
import { initWebGPU } from '../webgpu/init.js';

export class MatMul extends AutogradFunction {
  static async forward(ctx: Context, a: Tensor, b: Tensor) {
    const device = await initWebGPU();

    // Load the shader code
    const response = await fetch('/src/shaders/matmul.wgsl');
    const shaderCode = await response.text();

    // Create shader module
    const module = device.createShaderModule({
      code: shaderCode,
    });

    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });

    // Create uniform buffer for dimensions
    const uniformBuffer = device.createBuffer({
      size: 3 * 4, // 3 u32 values
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Write dimensions to uniform buffer
    const uniformTypedArray = new Uint32Array([a.shape[0], a.shape[1], b.shape[1]]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformTypedArray);

    const bufferA = device.createBuffer({
      size: a.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bufferB = device.createBuffer({
      size: b.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const resultBuffer = device.createBuffer({
      size: (a.shape[0] * b.shape[1]) * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    device.queue.writeBuffer(bufferA, 0, a.data);
    device.queue.writeBuffer(bufferB, 0, b.data);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufferA } },
        { binding: 2, resource: { buffer: bufferB } },
        { binding: 3, resource: { buffer: resultBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(a.shape[0] / 8),
      Math.ceil(b.shape[1] / 8)
    );
    pass.end();

    // Create a staging buffer to read the results
    const stagingBuffer = device.createBuffer({
      size: (a.shape[0] * b.shape[1]) * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    encoder.copyBufferToBuffer(
      resultBuffer,
      0,
      stagingBuffer,
      0,
      stagingBuffer.size
    );

    // Finish encoding and submit the commands
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    // Read the results
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultArray = new Float32Array(stagingBuffer.getMappedRange());
    const resultCopy = new Float32Array(resultArray);
    stagingBuffer.unmap();

    const resultTensor = new Tensor(resultCopy, [a.shape[0], b.shape[1]], true);

    ctx.save_for_backward(a, b);

    return resultTensor;
  }

  static async backward(ctx: Context, grad_output: Tensor) {
    const [a, b] = ctx.saved_tensors;

    const grad_a = await MatMul.forward(ctx, grad_output, b.transpose());
    const grad_b = await MatMul.forward(ctx, a.transpose(), grad_output);

    return [grad_a, grad_b];
  }
}