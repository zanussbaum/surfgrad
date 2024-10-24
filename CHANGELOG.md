# Changelog

## [1.0.2](https://github.com/zanussbaum/surfgrad/compare/v1.0.1...v1.0.2) (2024-10-24)


### Bug Fixes

* don't run ci tests on release please merge ([8d36f68](https://github.com/zanussbaum/surfgrad/commit/8d36f68f3f4a965c2d99433e6e1582e056a7e8f2))

## [1.0.1](https://github.com/zanussbaum/surfgrad/compare/v1.0.0...v1.0.1) (2024-10-24)


### Bug Fixes

* does this rerun release-please ([a9c3f72](https://github.com/zanussbaum/surfgrad/commit/a9c3f72a6bcd283d038626c599eef18bc895c98b))

## 1.0.0 (2024-10-24)


### Features

* 1.6TFLOPS matmul ([71b159c](https://github.com/zanussbaum/surfgrad/commit/71b159c672c606bb1be4530bc61fa22763c6556d))
* 2d tiling 360 GFLOP/s ([ae66be6](https://github.com/zanussbaum/surfgrad/commit/ae66be6362a8a785d7080bf9438ce97534793611))
* 2d tiling unrolled ([fd55984](https://github.com/zanussbaum/surfgrad/commit/fd5598432dddd3d436caccd799875bc0a967419b))
* 4k matmul and 280 GFLOP/s ([3fb91ba](https://github.com/zanussbaum/surfgrad/commit/3fb91ba79d19010c1ebe998697e001f5ea14c482))
* 8 results per thread is 150 GFLOP/s ([287283a](https://github.com/zanussbaum/surfgrad/commit/287283a94a2439f64b0be25324e4fde64fefaf9f))
* 8x8 2d tile 1TFLOP! ([27ef53f](https://github.com/zanussbaum/surfgrad/commit/27ef53f043d66d21c9759eb98314d47ea15ae9fd))
* ability to run benchmarks ([4326783](https://github.com/zanussbaum/surfgrad/commit/4326783af05f682e44a42de393aaa40c20300727))
* add op ([b7bbd91](https://github.com/zanussbaum/surfgrad/commit/b7bbd91f45e7801c50d8a21c4929366fdf469aa7))
* added unit tests ([849d359](https://github.com/zanussbaum/surfgrad/commit/849d3594895c4b8e95958325c64400457d09fc9b))
* calculate 1d 8 results per thread (300 GFLOP/s) at 2048 but slower at 4k ([c12b269](https://github.com/zanussbaum/surfgrad/commit/c12b269a4b3bb02c096e6eba784fe3f36cdeb7b5))
* elementwise (broadcastable) multiplication ([9c2878a](https://github.com/zanussbaum/surfgrad/commit/9c2878abcff40e7f92e61a3058fc1770f6135c54))
* elementwise mul (only one requires grad) ([5eea14f](https://github.com/zanussbaum/surfgrad/commit/5eea14fa25f282fe38f2bb11def1ca9d235fcfc8))
* ln log exp ops ([a76bab7](https://github.com/zanussbaum/surfgrad/commit/a76bab75a902be59a95d581797cb1023ad897e2c))
* ln op ([c3b6e23](https://github.com/zanussbaum/surfgrad/commit/c3b6e2369030bee677f3149a99b7d256ed2b06b9))
* ln op ([324104f](https://github.com/zanussbaum/surfgrad/commit/324104f3040b0c796dfbf3244afa5a7f20214e66))
* logo ([6892680](https://github.com/zanussbaum/surfgrad/commit/689268050a94ae62d825659517eb4922166a6f45))
* refactor boilerplate + exp,log ([5a7e7a6](https://github.com/zanussbaum/surfgrad/commit/5a7e7a6fdc749e6e6837905bc2e6e607b98decac))
* relu op, refactor to actually return grad_output * grad ([6aec3bd](https://github.com/zanussbaum/surfgrad/commit/6aec3bd2b57643771fafb41fcb91df42c1add20a))
* throw error if wrong shapes ([7c2c668](https://github.com/zanussbaum/surfgrad/commit/7c2c66822c1410f7de554ac85563c74492273733))
* til you can label buffers! ([bd77b7a](https://github.com/zanussbaum/surfgrad/commit/bd77b7a56c3dbd9242f7b548cf340c49b2f237f6))
* working "single" thread matmul ([0cb3aa8](https://github.com/zanussbaum/surfgrad/commit/0cb3aa86aca1b55b894190623322b0b86e3f6858))
* working autograd! now need to clean up interface ([b5796d3](https://github.com/zanussbaum/surfgrad/commit/b5796d3d06487365fff5032867ae447b5f08c2a8))
* working demo shaders ! ([e44fbbc](https://github.com/zanussbaum/surfgrad/commit/e44fbbc2cf1a61a0f04adf226707f9b55a20dd17))
* working matmul forward backward ([9c127bd](https://github.com/zanussbaum/surfgrad/commit/9c127bd2d43227c060cb3eb9e0ab702c5bc9b1fa))


### Bug Fixes

* (slower) shared memory doesn't seem to help ([1aa1a4e](https://github.com/zanussbaum/surfgrad/commit/1aa1a4e3efedbc9b676eb90381af8a26130e3d37))
* action version? ([cf0eec2](https://github.com/zanussbaum/surfgrad/commit/cf0eec2c2f1faae2a29a17e3fbbefe7cf1f8a166))
* actually calculate past 256 ! ([08219c7](https://github.com/zanussbaum/surfgrad/commit/08219c713f61ef64f457c73a6b161b11a6f6a483))
* actually run benchmark ([8f387c2](https://github.com/zanussbaum/surfgrad/commit/8f387c26f3179e231f693615733f5404ed6df098))
* get branch name (for release-pleas) ([f5f42f1](https://github.com/zanussbaum/surfgrad/commit/f5f42f1735663a9a3953f6581b8e235fac44c198))
* iterating on matmul, can do 4096, but only 50GFLOP/s ([04ce98a](https://github.com/zanussbaum/surfgrad/commit/04ce98a3d7e36300699b877cad5d91481951e29a))
* license ([102cdbe](https://github.com/zanussbaum/surfgrad/commit/102cdbef7c908104f3ff0312336ad3cacb21a42f))
* use secret ([5d15211](https://github.com/zanussbaum/surfgrad/commit/5d152110632c6bbbe3fa69a8180eddc9ba14171a))
* version ([b4964bb](https://github.com/zanussbaum/surfgrad/commit/b4964bb81d4e319e5e4e00fb95f02a751ae5d59d))
* workgroup size (doesn't do much) still at 1.5 TFLOPS ([d06fb99](https://github.com/zanussbaum/surfgrad/commit/d06fb9958451b53174947c21059006c1cff8b06d))
