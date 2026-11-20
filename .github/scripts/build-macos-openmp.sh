#!/usr/bin/env bash
set -euo pipefail

# Build libomp for the wheel's minimum OS. A Homebrew bottle targets the host OS.
version=20.1.8
release_url="https://github.com/llvm/llvm-project/releases/download/llvmorg-$version"
work_dir="$(mktemp -d "${RUNNER_TEMP:?}/omnigbdt-openmp.XXXXXXXX")"
package_dir="$PWD/omnigbdt"
cd "$work_dir"

for project in openmp cmake; do
    curl --fail --location --silent --show-error --retry 3 \
        "$release_url/$project-$version.src.tar.xz" -o "$project.tar.xz"
done
shasum -a 256 -c <<'CHECKSUMS'
b21c04ee9cbe56e200c5d83823765a443ee6389bbc3f64154c96e94016e6cee9  openmp.tar.xz
3319203cfd1172bbac50f06fa68e318af84dcb5d65353310c0586354069d6634  cmake.tar.xz
CHECKSUMS
for project in openmp cmake; do
    mkdir "$project"
    tar -xf "$project.tar.xz" --strip-components=1 -C "$project"
done

cmake -S openmp -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:?}" \
    -DCMAKE_INSTALL_PREFIX="${CMAKE_PREFIX_PATH:?}" \
    -DCMAKE_INSTALL_NAME_DIR="$CMAKE_PREFIX_PATH/lib" \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF -DOPENMP_ENABLE_OMPT_TOOLS=OFF
cmake --build build --parallel 3
cmake --install build
cp openmp/LICENSE.TXT "$package_dir/LICENSE.openmp.txt"
