project('OceanBase_CE', ['c', 'cpp'],
  version: '4.3.5.2',
  #'description', 'oceanbase distributed database system', 'homepage_url', 'https://open.oceanbase.com/', 'languages', 'cpp', 'c', 'asm',
  default_options : ['default_library=static'])
  message(['STATUS', 'open source build enabled'])

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

prefix = get_option('prefix')
buildtype = get_option('buildtype')

# Dependencies
openssl_req = '>= 1.0'
openssl_dep = dependency('openssl', version: openssl_req)

easy_deps = [
  openssl_dep,
]

libm_dep = cc.find_library('m', required: false)
if libm_dep.found()
  easy_deps += [libm_dep]
endif

libpthread_dep = cc.find_library('pthread', required: false)
if libpthread_dep.found()
  easy_deps += [libpthread_dep]
endif

if get_option('compile-dll-mode')
  add_project_arguments(
    [ '-DCOMPILE_DLL_MODE'],
    language: 'c')
  add_project_arguments(
    [ '-DCOMPILE_DLL_MODE'],
    language: 'cpp')
endif

#if 'ENABLE_COMPILE_DLL_MODE'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DCOMPILE_DLL_MODE'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DCOMPILE_DLL_MODE'
#endif
#if 'OB_ERRSIM'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DERRSIM'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DERRSIM'
#  ob_trans_errsim = 'ON'
#endif
#if 'OB_DIS_SEARRAY'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DDISABLE_SE_ARRAY'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DDISABLE_SE_ARRAY'
#endif
#if 'OB_TRANS_ERRSIM'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DTRANS_ERRSIM'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DTRANS_ERRSIM'
#endif
#if 'TRANS_MODULE_TEST'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DTRANS_MODULE_TEST'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DTRANS_MODULE_TEST'
#endif
#if 'ENABLE_LATCH_DIAGNOSE'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DENABLE_LATCH_DIAGNOSE'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DENABLE_LATCH_DIAGNOSE'
#endif
#if 'ENABLE_SMART_VAR_CHECK'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DENABLE_SMART_VAR_CHECK'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DENABLE_SMART_VAR_CHECK'
#endif
#if 'ENABLE_DEBUG_LOG'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DENABLE_DEBUG_LOG'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DENABLE_DEBUG_LOG'
#  if 'NOT', 'DEFINED', 'ENABLE_SERIALIZATION_CHECK'
#    enable_serialization_check = enable_debug_log
#  endif
#endif
#if 'ENABLE_SERIALIZATION_CHECK'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DENABLE_SERIALIZATION_CHECK'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DENABLE_SERIALIZATION_CHECK'
#endif
#if 'OB_GPERF_MODE'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -D__NEED_PERF__'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -D__NEED_PERF__'
#endif
#if 'ENABLE_PERF_MODE'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DPERF_MODE'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DPERF_MODE'
#endif
#if 'ENABLE_MEMORY_DIAGNOSIS'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DMEMORY_DIAGNOSIS=1'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DMEMORY_DIAGNOSIS=1'
#endif
#if 'ENABLE_OBJ_LEAK_CHECK'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DENABLE_OBJ_LEAK_CHECK=1'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DENABLE_OBJ_LEAK_CHECK=1'
#endif
#if 'ENABLE_FATAL_ERROR_HANG'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DFATAL_ERROR_HANG'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DFATAL_ERROR_HANG'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DENABLE_500_MEMORY_LIMIT'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DENABLE_500_MEMORY_LIMIT'
#endif
#if 'DETECT_RECURSION'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DDETECT_RECURSION'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DDETECT_RECURSION'
#endif
#if 'OB_USE_ASAN'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DOB_USE_ASAN'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DOB_USE_ASAN'
#endif
#if 'ENABLE_SANITY'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DENABLE_SANITY'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DENABLE_SANITY'
#endif
#if 'OB_BUILD_PACKAGE'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DOB_BUILD_PACKAGE'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DOB_BUILD_PACKAGE'
#endif
#if 'OB_ENABLE_MCMODEL'
#  if 'CMAKE_SYSTEM_PROCESSOR', 'STREQUAL', 'x86_64'
#    cmake_c_flags = '${CMAKE_C_FLAGS} -mcmodel=large'
#    cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -mcmodel=large'
#  endif
#endif
## message(['STATUS', 'This is BINARY dir ', project_binary_dir])
## message(['STATUS', 'This is SOURCE dir ', project_source_dir])
#cmake_position_independent_code = 'ON'
#hyperscan_lib = ''
#if architecture, 'STREQUAL', 'x86_64'
#  hyperscan_lib = '${DEP_DIR}/lib/libhs.a'
#endif
#if 'EXISTS', cmake_source_dir, '/close_modules'
#  # message(['STATUS', 'will add close_modules sub directory'])
#  subdir('close_modules')
#endif

# Paths
top_srcdir = include_directories('.')
top_src_srcdir = include_directories('./src')

srcdir = meson.current_source_dir()
builddir = meson.current_build_dir()

subdir('deps/easy')
subdir('deps/oblib')
#subdir('src/objit')
#subdir('src')
