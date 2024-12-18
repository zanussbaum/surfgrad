export const expShader = `
struct Dimensions {
  M: u32,
  N: u32,
}

@group(0) @binding(0) var<uniform> dimensions: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global_idx = global_id.x;
  let row = global_idx / dimensions.N;
  let col = global_idx % dimensions.N;

  if (global_idx < dimensions.M * dimensions.N) {
    result[row * dimensions.N + col] = exp(a[row * dimensions.N + col]);
  }
}
`;
