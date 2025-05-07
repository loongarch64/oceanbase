project('OceanBase_CE', ['c', 'cpp'],
  version: '4.3.5.2',
  default_options : ['default_library=static', 'warning_level=3'],
  meson_version: '>= 1.0.0',
  )
message(['STATUS', 'open source build enabled'])

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

prefix = get_option('prefix')
buildtype = get_option('buildtype')

filter_obj = false
objcopy = find_program('objcopy', native: true)
if cc.get_id() == 'clang'
  message('allow_obj_filter doesn\'t work with clang due to it not supporting pre-linking')
endif
if get_option('default_library') != 'shared' and not get_option('b_lto')
  if objcopy.found()
    message('Using objcopy to localize symbols in static library')
    filter_obj = true
  else
    error('objcopy not found, unable to localize symbols in static library')
  endif
endif

# Dependencies
openssl_req = '>= 1.1.1'
apr_req = '>= 1.6.5'
cos_c_sdk_req = '>= 5.0.21'
mxml_req = '>= 2.12.0'
aws_sdk_cpp_req = '>= 1.11.156'
lua_req = '>= 5.4.3'

libm = cc.find_library('m', required : true)
libpthread = cc.find_library('pthread', required : true)
openssl_dep = dependency('openssl', version: openssl_req)
apr_dep = dependency('apr-1', version: apr_req)
cos_c_sdk_dep = dependency('cos-c-sdk', version: cos_c_sdk_req)
aws_sdk_cpp_dep = dependency('aws-cpp-sdk-s3', version: aws_sdk_cpp_req)
lua_dep = dependency('lua', version: lua_req)

#add_project_arguments([
#  '-D_NO_EXCEPTION',
#  ], language: ['c', 'cpp'])

if cc.get_id() == 'clang'
  message('is clang')
  add_project_arguments([
    '-Wno-reserved-user-defined-literal',
    ], language: 'cpp')
else
  message('is not clang')
endif

#if get_option('cpp-standard-20')
#  add_project_arguments([
#    '-std=gnu++20',
#    '-DCPP_STANDARD_20'
#    ], language: 'cpp')
#else
#  add_project_arguments([
#    '-std=gnu++11',
#    ], language: 'cpp')
#endif

if get_option('compile-dll-mode')
  add_project_arguments(
    [ '-DCOMPILE_DLL_MODE'],
    language: ['c', 'cpp']
    )
endif

#if 'ENABLE_COMPILE_DLL_MODE'
#  cmake_c_flags = '${CMAKE_C_FLAGS} -DCOMPILE_DLL_MODE'
#  cmake_cxx_flags = '${CMAKE_CXX_FLAGS} -DCOMPILE_DLL_MODE'
#endif

if get_option('ob_errsim')
  add_project_arguments(
    ['-DERRSIM'],
    language : ['c', 'cpp'],
    )
endif

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
top_sdir = include_directories('.')
top_src_sdir = include_directories('./src')

cur_sdir = meson.current_source_dir()
cur_bdir = meson.current_build_dir()

subdir('deps/easy')
subdir('deps/oblib')
#subdir('src/objit')
#subdir('src')
