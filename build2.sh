#!/bin/bash
source /etc/os-release

case "$ID" in
    arch)
        # Archlinux:
        # yay -S rapidjson openssl-1.0 mariadb isa-l
        #export PKG_CONFIG_PATH=/usr/lib/openssl-1.0/pkgconfig:$PKG_CONFIG_PATH
        ;;
esac

cmake_flags=(
    #-DCMAKE_BUILD_TYPE=Release # Why Release build failed?
    -DCMAKE_BUILD_TYPE=Debug
    -DCMAKE_VERBOSE_MAKEFILE=ON

    -DOB_BUILD_RPM=OFF

    -DOB_USE_CLANG=OFF
    -DOB_USE_LLVM_LIBTOOLS=OFF
    -DOB_COMPRESS_DEBUG_SECTIONS=OFF
    -DOB_STATIC_LINK_LGPL_DEPS=OFF

    -DOB_USE_CCACHE=OFF
    -DOB_ENABLE_PCH=ON
    -DOB_ENALBE_UNITY=ON
    -DOB_MAX_UNITY_BATCH_SIZE=30
    -DOB_USE_ASAN=OFF

    -DOB_RELEASEID=1
)

mkdir -p build
cd build

cmake \
    "${cmake_flags[@]}" \
    ..

make -j`nproc`
