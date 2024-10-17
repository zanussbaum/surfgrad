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
        for (var i = 0u; i < TILE_M; i++) {
            let a_element = a[(row + i) * dimensions.K + k];
            for (var j = 0u; j < TILE_N; j++) {
                let b_element = b[k * dimensions.N + (col + j)];
                sums[i][j] += a_element * b_element;
            }
        }
    }

    // Write results
    for (var i = 0u; i < TILE_M; i++) {
        for (var j = 0u; j < TILE_N; j++) {
            let output_row = row + i;
            let output_col = col + j;
            if (output_row < dimensions.M && output_col < dimensions.N) {
                result[output_row * dimensions.N + output_col] = sums[i][j];
            }
        }
    }
}