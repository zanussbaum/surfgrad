struct Dimensions {
  M: u32,
  K: u32,
  N: u32,
}

@group(0) @binding(0) var<uniform> dimensions: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(2, 2)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;

  if (row < dimensions.M && col < dimensions.N) {
    var sum : f32 = 0.0;
    for (var i: u32 = 0u; i < dimensions.K; i = i + 1u) {
      sum = sum + a[row * dimensions.K + i] * b[i * dimensions.N + col];
    }
    result[row * dimensions.N + col] = sum;
  }
}