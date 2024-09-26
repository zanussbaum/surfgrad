async function initWebGPU() {
    const gpu = navigator.gpu as GPU;
    if (!gpu) {
        throw new Error("WebGPU not supported on this browser.");
    }

    const adapter = await gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice();
    const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
    const context = canvas.getContext("webgpu") as GPUCanvasContext;

    const format = gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: format,
    });

    // Load the shader code from the file
    const shader = await fetch("src/shaders/shader.wgsl").then(res => res.text());

    // Create a shader module
    const shaderModule = device.createShaderModule({
        code: shader,
    });

    // Create a pipeline layout (empty for now)
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [],
    });

    // Define the render pipeline
    const renderPipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vs_main', // Entry point for the vertex shader
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fs_main', // Entry point for the fragment shader
            targets: [
                {
                    format: format,
                },
            ],
        },
        primitive: {
            topology: 'line-strip', // Define how the vertices are connected
        },
    });

    // Create a command encoder for rendering
    const commandEncoder = device.createCommandEncoder();

    // Create a render pass descriptor
    const renderPassDescriptor = {
        colorAttachments: [
            {
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 }, // Background color (black)
                loadOp: 'clear' as GPULoadOp,
                storeOp: 'store' as GPUStoreOp,
            },
        ],
    };

    // Begin the render pass
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(renderPipeline);
    passEncoder.draw(3, 1, 0, 0); // Draw 3 vertices (a triangle)
    passEncoder.end();

    // Finish and submit the command buffer
    const commandBuffer = commandEncoder.finish();
    device.queue.submit([commandBuffer]);

    console.log("WebGPU initialized and shader rendered successfully!");
}

initWebGPU().catch(console.error);
