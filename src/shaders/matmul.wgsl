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
       workgroup_index * (BLOCKSIZE * BLOCKSIZE) +
       local_invocation_index;

  let row = index / dimensions.N;
  let col = index % dimensions.N;

  if (index < dimensions.M * dimensions.N) {
    var sum : f32 = 0.0;
    for (var i: u32 = 0u; i < dimensions.K; i = i + 1u) {
      sum = sum + a[row * dimensions.K + i] * b[i * dimensions.N + col];
    }
    result[index] = sum;
  }
}