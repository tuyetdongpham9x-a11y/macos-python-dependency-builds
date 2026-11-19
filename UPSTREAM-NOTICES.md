# Upstream licenses and provenance

The recipe downloads upstream bytes at the exact commits or SHA-256 hashes in `public-lock.json`. No upstream source is vendored in this repository and no patches are included.

- [PyTorch source license at the pinned commit](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/LICENSE). The package includes additional third-party notices; its own build metadata collects license files into the wheel. The recipe copies the wheel's original license/notice files into evidence without replacing them.
- [TorchVision source license at the pinned commit](https://github.com/pytorch/vision/blob/8fb87713a24951e639c494b0f2a8a81b5f8e33a6/LICENSE).
- [CPython source license](https://docs.python.org/3/license.html) and [python-build-standalone licensing information](https://gregoryszorc.com/docs/python-build-standalone/main/distributions.html). The pinned standalone interpreter is used only as a temporary build/probe bootstrap and is not included in the optional wheel artifact.
- [CMake 4.1.2 package provenance and files](https://pypi.org/project/cmake/4.1.2/#files) and [Ninja 1.13.0 package provenance and files](https://pypi.org/project/ninja/1.13.0/#files). These pinned build tools are temporary build inputs and are not redistributed in the wheel artifact.
- NumPy, Pillow, setuptools, wheel, Cython and the other explicitly pinned bootstrap packages retain their respective upstream licenses. They are temporary build/probe inputs. Public library-wheel license notices are retained when sharing generated or official library wheels.

The MIT license in this repository does not relicense third-party dependencies. A checksum identifies bytes; it does not establish a successful native build or compatibility. The actual CPU ABI report is separate from the checksum report.

- [ccache4.14.3 source and GPL-3.0-or-later license](https://github.com/ccache/ccache/blob/v4.14.3/GPL-3.0.txt). The hash-pinned public Darwin build tool is temporary and is not redistributed in the wheel/evidence artifact. Its compiler-cache output contains only the public dependency compilation entries described in the recipe.
