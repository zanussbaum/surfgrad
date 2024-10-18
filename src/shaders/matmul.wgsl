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
const TILE_M: u32 = 8;  // Tile size in M dimension
const TILE_N: u32 = 8;  // Tile size in N dimension

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
      let a04 = a[(row + 4) * dimensions.K + k];
      let a05 = a[(row + 5) * dimensions.K + k];
      let a06 = a[(row + 6) * dimensions.K + k];
      let a07 = a[(row + 7) * dimensions.K + k];
      let b_00 = b[k * dimensions.N + col];
      let b01 = b[k * dimensions.N + (col + 1)];
      let b02 = b[k * dimensions.N + (col + 2)];
      let b03 = b[k * dimensions.N + (col + 3)];
      let b04 = b[k * dimensions.N + (col + 4)];
      let b05 = b[k * dimensions.N + (col + 5)];
      let b06 = b[k * dimensions.N + (col + 6)];
      let b07 = b[k * dimensions.N + (col + 7)];
      sums[0][0] += a_00 * b_00;
      sums[0][1] += a_00 * b01;
      sums[0][2] += a_00 * b02;
      sums[0][3] += a_00 * b03;
      sums[0][4] += a_00 * b04;
      sums[0][5] += a_00 * b05;
      sums[0][6] += a_00 * b06;
      sums[0][7] += a_00 * b07;
      sums[1][0] += a01 * b_00;
      sums[1][1] += a01 * b01;
      sums[1][2] += a01 * b02;
      sums[1][3] += a01 * b03;
      sums[1][4] += a01 * b04;
      sums[1][5] += a01 * b05;
      sums[1][6] += a01 * b06;
      sums[1][7] += a01 * b07;
      sums[2][0] += a02 * b_00;
      sums[2][1] += a02 * b01;
      sums[2][2] += a02 * b02;
      sums[2][3] += a02 * b03;
      sums[2][4] += a02 * b04;
      sums[2][5] += a02 * b05;
      sums[2][6] += a02 * b06;
      sums[2][7] += a02 * b07;
      sums[3][0] += a03 * b_00;
      sums[3][1] += a03 * b01;
      sums[3][2] += a03 * b02;
      sums[3][3] += a03 * b03;
      sums[3][4] += a03 * b04;
      sums[3][5] += a03 * b05;
      sums[3][6] += a03 * b06;
      sums[3][7] += a03 * b07;
      sums[4][0] += a04 * b_00;
      sums[4][1] += a04 * b01;
      sums[4][2] += a04 * b02;
      sums[4][3] += a04 * b03;
      sums[4][4] += a04 * b04;
      sums[4][5] += a04 * b05;
      sums[4][6] += a04 * b06;
      sums[4][7] += a04 * b07;
      sums[5][0] += a05 * b_00;
      sums[5][1] += a05 * b01;
      sums[5][2] += a05 * b02;
      sums[5][3] += a05 * b03;
      sums[5][4] += a05 * b04;
      sums[5][5] += a05 * b05;
      sums[5][6] += a05 * b06;
      sums[5][7] += a05 * b07;
      sums[6][0] += a06 * b_00;
      sums[6][1] += a06 * b01;
      sums[6][2] += a06 * b02;
      sums[6][3] += a06 * b03;
      sums[6][4] += a06 * b04;
      sums[6][5] += a06 * b05;
      sums[6][6] += a06 * b06;
      sums[6][7] += a06 * b07;
      sums[7][0] += a07 * b_00;
      sums[7][1] += a07 * b01;
      sums[7][2] += a07 * b02;
      sums[7][3] += a07 * b03;
      sums[7][4] += a07 * b04;
      sums[7][5] += a07 * b05;
      sums[7][6] += a07 * b06;
      sums[7][7] += a07 * b07;
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
        if (col + 4 < dimensions.N) {
            result[row * dimensions.N + (col + 4)] = sums[0][4];
        }
        if (col + 5 < dimensions.N) {
            result[row * dimensions.N + (col + 5)] = sums[0][5];
        }
        if (col + 6 < dimensions.N) {
            result[row * dimensions.N + (col + 6)] = sums[0][6];
        }
        if (col + 7 < dimensions.N) {
            result[row * dimensions.N + (col + 7)] = sums[0][7];
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
        if (col + 4 < dimensions.N) {
            result[(row + 1) * dimensions.N + (col + 4)] = sums[1][4];
        }
        if (col + 5 < dimensions.N) {
            result[(row + 1) * dimensions.N + (col + 5)] = sums[1][5];
        }
        if (col + 6 < dimensions.N) {
            result[(row + 1) * dimensions.N + (col + 6)] = sums[1][6];
        }
        if (col + 7 < dimensions.N) {
            result[(row + 1) * dimensions.N + (col + 7)] = sums[1][7];
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
        if (col + 4 < dimensions.N) {
            result[(row + 2) * dimensions.N + (col + 4)] = sums[2][4];
        }
        if (col + 5 < dimensions.N) {
            result[(row + 2) * dimensions.N + (col + 5)] = sums[2][5];
        }
        if (col + 6 < dimensions.N) {
            result[(row + 2) * dimensions.N + (col + 6)] = sums[2][6];
        }
        if (col + 7 < dimensions.N) {
            result[(row + 2) * dimensions.N + (col + 7)] = sums[2][7];
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
        if (col + 4 < dimensions.N) {
            result[(row + 3) * dimensions.N + (col + 4)] = sums[3][4];
        }
        if (col + 5 < dimensions.N) {
            result[(row + 3) * dimensions.N + (col + 5)] = sums[3][5];
        }
        if (col + 6 < dimensions.N) {
            result[(row + 3) * dimensions.N + (col + 6)] = sums[3][6];
        }
        if (col + 7 < dimensions.N) {
            result[(row + 3) * dimensions.N + (col + 7)] = sums[3][7];
        }
    }
    if (row + 4 < dimensions.M) {
        if (col < dimensions.N) {
            result[(row + 4) * dimensions.N + col] = sums[4][0];
        }
        if (col + 1 < dimensions.N) {
            result[(row + 4) * dimensions.N + (col + 1)] = sums[4][1];
        }
        if (col + 2 < dimensions.N) {
            result[(row + 4) * dimensions.N + (col + 2)] = sums[4][2];
        }
        if (col + 3 < dimensions.N) {
            result[(row + 4) * dimensions.N + (col + 3)] = sums[4][3];
        }
        if (col + 4 < dimensions.N) {
            result[(row + 4) * dimensions.N + (col + 4)] = sums[4][4];
        }
        if (col + 5 < dimensions.N) {
            result[(row + 4) * dimensions.N + (col + 5)] = sums[4][5];
        }
        if (col + 6 < dimensions.N) { 
            result[(row + 4) * dimensions.N + (col + 6)] = sums[4][6];
        }
        if (col + 7 < dimensions.N) {
            result[(row + 4) * dimensions.N + (col + 7)] = sums[4][7];
        }
    }
    if (row + 5 < dimensions.M) {
        if (col < dimensions.N) {
            result[(row + 5) * dimensions.N + col] = sums[5][0];
        }
        if (col + 1 < dimensions.N) {
            result[(row + 5) * dimensions.N + (col + 1)] = sums[5][1];
        }
        if (col + 2 < dimensions.N) {
            result[(row + 5) * dimensions.N + (col + 2)] = sums[5][2];
        }
        if (col + 3 < dimensions.N) {
            result[(row + 5) * dimensions.N + (col + 3)] = sums[5][3]; 
        }
        if (col + 4 < dimensions.N) {
            result[(row + 5) * dimensions.N + (col + 4)] = sums[5][4];
        }
        if (col + 5 < dimensions.N) {
            result[(row + 5) * dimensions.N + (col + 5)] = sums[5][5];
        }
        if (col + 6 < dimensions.N) {
            result[(row + 5) * dimensions.N + (col + 6)] = sums[5][6];
        }
        if (col + 7 < dimensions.N) {
            result[(row + 5) * dimensions.N + (col + 7)] = sums[5][7];
        }
    }
    if (row + 6 < dimensions.M) {
        if (col < dimensions.N) {
            result[(row + 6) * dimensions.N + col] = sums[6][0];
        }
        if (col + 1 < dimensions.N) {
            result[(row + 6) * dimensions.N + (col + 1)] = sums[6][1];
        }
        if (col + 2 < dimensions.N) {
            result[(row + 6) * dimensions.N + (col + 2)] = sums[6][2];
        }
        if (col + 3 < dimensions.N) {
            result[(row + 6) * dimensions.N + (col + 3)] = sums[6][3];
        }
        if (col + 4 < dimensions.N) {
            result[(row + 6) * dimensions.N + (col + 4)] = sums[6][4];
        }
        if (col + 5 < dimensions.N) {
            result[(row + 6) * dimensions.N + (col + 5)] = sums[6][5];
        }
        if (col + 6 < dimensions.N) {
            result[(row + 6) * dimensions.N + (col + 6)] = sums[6][6];
        }
        if (col + 7 < dimensions.N) {
            result[(row + 6) * dimensions.N + (col + 7)] = sums[6][7];
        }
    }
    if (row + 7 < dimensions.M) {
        if (col < dimensions.N) {
            result[(row + 7) * dimensions.N + col] = sums[7][0];
        }
        if (col + 1 < dimensions.N) {
            result[(row + 7) * dimensions.N + (col + 1)] = sums[7][1];
        }
        if (col + 2 < dimensions.N) {
            result[(row + 7) * dimensions.N + (col + 2)] = sums[7][2];
        }
        if (col + 3 < dimensions.N) {
            result[(row + 7) * dimensions.N + (col + 3)] = sums[7][3];
        }
        if (col + 4 < dimensions.N) {
            result[(row + 7) * dimensions.N + (col + 4)] = sums[7][4];
        }
        if (col + 5 < dimensions.N) {
            result[(row + 7) * dimensions.N + (col + 5)] = sums[7][5];
        }
        if (col + 6 < dimensions.N) {
            result[(row + 7) * dimensions.N + (col + 6)] = sums[7][6];
        }
        if (col + 7 < dimensions.N) {
            result[(row + 7) * dimensions.N + (col + 7)] = sums[7][7];
        }
    }
}