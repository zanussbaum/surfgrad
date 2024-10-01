struct Dimensions {
  M: u32,
  N: u32,
}

@group(0) @binding(0) var<uniform> dimensions: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> scalar: f32;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row: u32 = global_id.x;
  let col: u32 = global_id.y;

  if (row < dimensions.M && col < dimensions.N) {
    let index = row * dimensions.N + col;
    result[index] = a[index] * scalar;
  }
}