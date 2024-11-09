export async function timeAndExecute(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  buffer: GPUBuffer,
): Promise<[Float32Array, number]> {
  const start = performance.now();
  device.queue.submit([encoder.finish()]);

  await buffer.mapAsync(GPUMapMode.READ);
  const resultArray = new Float32Array(buffer.getMappedRange());
  const totalTime = performance.now() - start;

  return [resultArray, totalTime];
}
