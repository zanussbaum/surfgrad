struct Params {
  M: u32,
  N: u32,
  K: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const BLOCKSIZE: u32 = 16;
var<workgroup> As: array<f32, BLOCKSIZE * BLOCKSIZE>;
var<workgroup> Bs: array<f32, BLOCKSIZE * BLOCKSIZE>;

@compute @workgroup_size(BLOCKSIZE * BLOCKSIZE)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>
) {
  let cRow = group_id.y;
  let cCol = group_id.x;
  let threadCol = local_id.x % BLOCKSIZE;
  let threadRow = local_id.x / BLOCKSIZE;

  var tmp: f32 = 0.0;
  
  for (var bkIdx: u32 = 0u; bkIdx < params.K; bkIdx += BLOCKSIZE) {
    if (bkIdx + threadCol < params.K && cRow * BLOCKSIZE + threadRow < params.M) {
      As[threadRow * BLOCKSIZE + threadCol] = A[(cRow * BLOCKSIZE + threadRow) * params.K + bkIdx + threadCol];
    }
    if (bkIdx + threadRow < params.K && cCol * BLOCKSIZE + threadCol < params.N) {
      Bs[threadRow * BLOCKSIZE + threadCol] = B[(bkIdx + threadRow) * params.N + cCol * BLOCKSIZE + threadCol];
    }
    
    workgroupBarrier();

    for (var dotIdx: u32 = 0u; dotIdx < BLOCKSIZE && bkIdx + dotIdx < params.K; dotIdx++) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    
    workgroupBarrier();
  }

  let row = cRow * BLOCKSIZE + threadRow;
  let col = cCol * BLOCKSIZE + threadCol;
  if (row < params.M && col < params.N) {
    let index = row * params.N + col;
    C[index] = tmp;
  }
}