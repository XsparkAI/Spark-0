# StarVLA (vendored for Spark-0)

A trimmed, patched copy of [starVLA/starVLA](https://github.com/starVLA/starVLA) at commit `c00b8e1` (2026-07-20), MIT-licensed (see `LICENSE`, `CITATION.cff`).

Spark-0 patches:

- tactile encoder and fusion, contact alignment, DA3 spatial alignment;
- action queries, flexible views, real-time chunking (RTC);
- precomputed / streaming normalization statistics, mixture sampling fixes;
- ZeRO-3 safety and the NaN fuse.

Removed, compared with upstream:

- documentation and assets;
- Docker files;
- GR00T/ZMQ servers;
- examples for other robots and benchmarks.

The Spark-0 data registry lives in `examples/Spark0/train_files/data_registry/`.

Do not run this package directly. Use the Spark-0 entry points (`scripts/train.sh`, `scripts/serve.sh`) from the repository root, which set the environment the registry reads. Model weights go under `playground/Pretrained_models/` (see `scripts/download_*.sh`).
