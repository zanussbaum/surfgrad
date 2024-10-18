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
const TILE_M: u32 = 4;  // Tile size in M dimension
const TILE_N: u32 = 4;  // Tile size in N dimension

@compute @workgroup_size(BLOCKSIZE, BLOCKSIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y * TILE_M;
    let col = global_id.x * TILE_N;

    var sums: array<array<f32, TILE_N>, TILE_M>;
    for (var i = 0u; i < TILE_M; i++) {
        for (var j = 0u; j < TILE_N; j++) {
            sums[i][j] = 0.0;
        }
    }

    // Compute the 2D tile
    for (var k = 0u; k < dimensions.K; k++) {
      let a_00 = a[row * dimensions.K + k];
      let a01 = a[(row + 1) * dimensions.K + k];
      let a02 = a[(row + 2) * dimensions.K + k];
      let a03 = a[(row + 3) * dimensions.K + k];
      let b_00 = b[k * dimensions.N + col];
      let b01 = b[k * dimensions.N + (col + 1)];
      let b02 = b[k * dimensions.N + (col + 2)];
      let b03 = b[k * dimensions.N + (col + 3)];
      sums[0][0] += a_00 * b_00;
      sums[0][1] += a_00 * b01;
      sums[0][2] += a_00 * b02;
      sums[0][3] += a_00 * b03;
      sums[1][0] += a01 * b_00;
      sums[1][1] += a01 * b01;
      sums[1][2] += a01 * b02;
      sums[1][3] += a01 * b03;
      sums[2][0] += a02 * b_00;
      sums[2][1] += a02 * b01;
      sums[2][2] += a02 * b02;
      sums[2][3] += a02 * b03;
      sums[3][0] += a03 * b_00;
      sums[3][1] += a03 * b01;
      sums[3][2] += a03 * b02;
      sums[3][3] += a03 * b03;
    }

    // Row 0
    if (row < dimensions.M) {
        if (col < dimensions.N) {
            result[row * dimensions.N + col] = sums[0][0];
        }
        if (col + 1 < dimensions.N) {
            result[row * dimensions.N + (col + 1)] = sums[0][1];
        }
        if (col + 2 < dimensions.N) {
            result[row * dimensions.N + (col + 2)] = sums[0][2];
        }
        if (col + 3 < dimensions.N) {
            result[row * dimensions.N + (col + 3)] = sums[0][3];
        }
    }

    // Row 1
    if (row + 1 < dimensions.M) {
        if (col < dimensions.N) {
            result[(row + 1) * dimensions.N + col] = sums[1][0];
        }
        if (col + 1 < dimensions.N) {
            result[(row + 1) * dimensions.N + (col + 1)] = sums[1][1];
        }
        if (col + 2 < dimensions.N) {
            result[(row + 1) * dimensions.N + (col + 2)] = sums[1][2];
        }
        if (col + 3 < dimensions.N) {
            result[(row + 1) * dimensions.N + (col + 3)] = sums[1][3];
        }
    }

    // Row 2
    if (row + 2 < dimensions.M) {
        if (col < dimensions.N) {
            result[(row + 2) * dimensions.N + col] = sums[2][0];
        }
        if (col + 1 < dimensions.N) {
            result[(row + 2) * dimensions.N + (col + 1)] = sums[2][1];
        }
        if (col + 2 < dimensions.N) {
            result[(row + 2) * dimensions.N + (col + 2)] = sums[2][2];
        }
        if (col + 3 < dimensions.N) {
            result[(row + 2) * dimensions.N + (col + 3)] = sums[2][3];
        }
    }

    // Row 3
    if (row + 3 < dimensions.M) {
        if (col < dimensions.N) {
            result[(row + 3) * dimensions.N + col] = sums[3][0];
        }
        if (col + 1 < dimensions.N) {
            result[(row + 3) * dimensions.N + (col + 1)] = sums[3][1];
        }
        if (col + 2 < dimensions.N) {
            result[(row + 3) * dimensions.N + (col + 2)] = sums[3][2];
        }
        if (col + 3 < dimensions.N) {
            result[(row + 3) * dimensions.N + (col + 3)] = sums[3][3];
        }
    }
}