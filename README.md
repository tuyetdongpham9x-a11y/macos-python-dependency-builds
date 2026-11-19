# Pinned public CPU wheel experiment

This repository contains only a generic build recipe, public dependency URLs and checksums, and a small tensor ABI probe. It has no application sources, commercial product branding, workers, model checkpoints, activation material, private endpoints, private repositories or credentials. No other repository or application checkout is accepted as input.

The manual workflow uses standard `macos-15-intel` or `macos-15` runners and refuses private repositories before allocating a runner. [GitHub documents standard public-repository runner compute as free and unlimited](https://docs.github.com/en/actions/reference/runners/github-hosted-runners). Artifact storage is a separate account quota: use `artifact_mode=none` if free storage is unavailable and keep paid spending disabled. No larger, GPU, paid or self-hosted runner is selected, and no caches or release uploads are created.

## Exact experiment

- Intel: build the unmodified public PyTorch 2.13.0 and TorchVision 0.28.0 source commits in `public-lock.json`, with CPU-only controls and Apple's Accelerate BLAS. Keep the compiler's standard baseline with `USE_NATIVE_ARCH=0` and an explicit thin `CMAKE_OSX_ARCHITECTURES=x86_64` target. No global `CFLAGS` or `CXXFLAGS` processor flags are forwarded. CUDA, MPS, OpenMP, MKL and distributed backends are disabled; image/video codecs in TorchVision are disabled. This is a deliberately scoped CPU dependency experiment.
- ARM: install the already public hash-pinned ARM wheels and run the same CPU ABI probe. This does not rebuild or claim CPU-only contents for the official ARM wheel, which may include MPS support.
- Both: use the hash-pinned CPython 3.11.18 bootstrap, NumPy 2.2.6 and explicit offline build/runtime dependencies. Public CMake 4.1.2 and Ninja 1.13.0 wheels are pinned by SHA-256 too. Pip never resolves extra packages or rewrites package metadata.
- Force `ATEN_CPU_CAPABILITY=default` for the probe and verify Intel reports the generic fallback. Probe only tensor/NumPy interchange, FFT, a small linear-algebra kernel and native TorchVision NMS. It downloads no model weights and proves no application inference, separation, GPU, installer or customer compatibility.

The Torch source is pinned to `cf30153c4c131c8164ee7798e5022d810682e2cb`; Vision to `8fb87713a24951e639c494b0f2a8a81b5f8e33a6`. Recursive Torch submodules follow immutable gitlinks, never `--remote`. No source patches or version-metadata workarounds are applied.

The pinned Torch CMake source deliberately builds a universal Intel/ARM host `protoc` before the thin target. The first Intel trial reached that bootstrap and stopped while finding Threads. Global Intel `-march` flags can invalidate its ARM compiler probes, so this recipe removes those global flags without changing upstream sources. This explains a concrete compiler incompatibility; a successful native Threads/protoc build is still unverified until the next run.

After a successful Torch wheel build, the actual main `build/CMakeCache.txt` must confirm the requested CPU-only flags, Accelerate BLAS, thin Intel architecture, macOS 15 target and absence of global processor-specific flags. A platform-capability message such as “MPS supported” is not used as proof that MPS is enabled or disabled. The final installed-library CPU ABI probe remains required before wheel sharing.

On configuration/build failure, capture only generated CMake caches and configure/error/output logs under the known `build` and `build_host_protoc` directories. Collection is bounded to 12 files, 1 MiB per file, 6 MiB aggregate and a three-second/5,000-directory scan budget. Larger files retain bounded head/tail excerpts. These diagnostics fit within the existing evidence cap; source trees, environment dumps and build binaries are never uploaded.

## Manual operation

After reviewing this exact public scope, run **Public native CPU wheel experiment** manually. Choose `x86_64` for the source build or `arm64` for the public-wheel ABI probe. This trial branch additionally accepts one exact-path push on `ci/intel-cpu-source-20261119-01`; the default branch retains the reviewed manual-only workflow. The push profile selects Intel and bounded wheel sharing. There is no schedule or pull-request trigger and no secrets input.

The job hardcap is 180 minutes; the driver stops after at most 170 minutes. Only two build workers are used. A standard runner has limited memory and disk: the driver refuses to start below 8 GiB free, but the complete source build may still exceed disk, memory or time and fail. Successful Intel compilation has **not been established** by preparation of this recipe. Clang, Xcode and SDK versions are recorded from the moving runner image; byte-identical wheel reproduction across image changes is not promised.

The default artifact is a closed text-evidence snapshot capped at **20 MiB total**, retained for one day. `wheels` is optional: the two library wheels, their original license notices, checksum list and evidence share a strict **256 MiB aggregate cap**. Wheels are shared only after both wheel checks and the generic native CPU ABI probe pass. Failed or partial builds retain text evidence only. If the cap is exceeded, no artifact is uploaded; wheel sizes are not assumed in advance. `none` uploads nothing. Neither the entire source checkout, Python installation, build tree, download cache nor any home directory is uploaded.

Every library wheel is checked for safe ZIP paths, CRC, bounded expanded size, native Mach-O architecture and original upstream license notices. Generated wheel hashes and actual compiler versions are recorded. A timeout stops/reaps the direct subprocess parent only; compiler descendants can remain until normal runner teardown. There are no process-group signals, cleanup retries, privilege changes or paid-runner fallbacks.

This recipe's MIT license applies only to these recipe/probe files. The dependencies keep their original licenses; see [UPSTREAM-NOTICES.md](UPSTREAM-NOTICES.md). Nothing here grants a license to unrelated applications or private assets.
