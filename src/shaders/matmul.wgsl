struct Dimensions {
  M: u32,
  K: u32,
  N: u32,
}

@group(0) @binding(0) var<uniform> dimensions: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

const BLOCKSIZE: u32 = 16;
const TILESIZE: u32 = 8;
@compute @workgroup_size(BLOCKSIZE, BLOCKSIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_index) local_invocation_index: u32,
        @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let workgroup_index =  
       workgroup_id.x +
       workgroup_id.y * num_workgroups.x +
       workgroup_id.z * num_workgroups.x * num_workgroups.y;

  let index =
       (workgroup_index * (BLOCKSIZE * BLOCKSIZE) +
       local_invocation_index) * TILESIZE;


  let row = index / dimensions.N;
  let col = index % dimensions.N;

  if (index < dimensions.M * dimensions.N) {
    var sum00: f32 = 0.0;
    var sum01: f32 = 0.0;
    var sum02: f32 = 0.0;
    var sum03: f32 = 0.0;
    var sum04: f32 = 0.0;
    var sum05: f32 = 0.0;
    var sum06: f32 = 0.0;
    var sum07: f32 = 0.0;

    for (var i: u32 = 0u; i < dimensions.K; i = i + 1u) {
      sum00 = sum00 + a[row * dimensions.K + i] * b[i * dimensions.N + col];
      sum01 = sum01 + a[row * dimensions.K + i] * b[i * dimensions.N + col + 1u];
      sum02 = sum02 + a[row * dimensions.K + i] * b[i * dimensions.N + col + 2u];
      sum03 = sum03 + a[row * dimensions.K + i] * b[i * dimensions.N + col + 3u];
      sum04 = sum04 + a[row * dimensions.K + i] * b[i * dimensions.N + col + 4u];
      sum05 = sum05 + a[row * dimensions.K + i] * b[i * dimensions.N + col + 5u];
      sum06 = sum06 + a[row * dimensions.K + i] * b[i * dimensions.N + col + 6u];
      sum07 = sum07 + a[row * dimensions.K + i] * b[i * dimensions.N + col + 7u];
    }
    result[row * dimensions.N + col] = sum00;
    result[row * dimensions.N + col + 1u] = sum01;
    result[row * dimensions.N + col + 2u] = sum02;
    result[row * dimensions.N + col + 3u] = sum03;
    result[row * dimensions.N + col + 4u] = sum04;
    result[row * dimensions.N + col + 5u] = sum05;
    result[row * dimensions.N + col + 6u] = sum06;
    result[row * dimensions.N + col + 7u] = sum07;
  }
}