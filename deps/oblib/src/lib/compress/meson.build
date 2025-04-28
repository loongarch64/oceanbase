#oblib_add_library([
#'compress',
#'ob_compressor.cpp',
#'ob_compressor.h',
#'ob_compress_util.h',
#'ob_compressor_pool.cpp',
#'ob_compressor_pool.h',
#'ob_stream_compressor.cpp',
#'ob_stream_compressor.h',
#'none/ob_none_compressor.cpp',
#'none/ob_none_compressor.h',
#'snappy/ob_snappy_compressor.cpp',
#'snappy/ob_snappy_compressor.h',
#'zlib/ob_zlib_compressor.cpp',
#'zlib/ob_zlib_compressor.h',
#'zstd/ob_zstd_compressor.cpp',
#'zstd/ob_zstd_compressor.h',
#'zstd/ob_zstd_stream_compressor.cpp',
#'zstd/ob_zstd_stream_compressor.h',
#'zstd_1_3_8/ob_zstd_compressor_1_3_8.cpp',
#'zstd_1_3_8/ob_zstd_compressor_1_3_8.h',
#'zstd_1_3_8/ob_zstd_stream_compressor_1_3_8.cpp',
#'zstd_1_3_8/ob_zstd_stream_compressor_1_3_8.h',
#'zlib_lite/ob_zlib_lite_compressor.cpp',
#'zlib_lite/ob_zlib_lite_compressor.h'])

compress_sources = [
'ob_compressor.cpp',
'ob_compressor.h',
'ob_compress_util.h',
'ob_compressor_pool.cpp',
'ob_compressor_pool.h',
'ob_stream_compressor.cpp',
'ob_stream_compressor.h',
'none/ob_none_compressor.cpp',
'none/ob_none_compressor.h',
'snappy/ob_snappy_compressor.cpp',
'snappy/ob_snappy_compressor.h',
'zlib/ob_zlib_compressor.cpp',
'zlib/ob_zlib_compressor.h',
'zstd/ob_zstd_compressor.cpp',
'zstd/ob_zstd_compressor.h',
'zstd/ob_zstd_stream_compressor.cpp',
'zstd/ob_zstd_stream_compressor.h',
'zstd_1_3_8/ob_zstd_compressor_1_3_8.cpp',
'zstd_1_3_8/ob_zstd_compressor_1_3_8.h',
'zstd_1_3_8/ob_zstd_stream_compressor_1_3_8.cpp',
'zstd_1_3_8/ob_zstd_stream_compressor_1_3_8.h']

subdir('zstd')
subdir('zstd_1_3_8')
subdir('lz4')
subdir('snappy')
#subdir('zlib_lite')

## target_link_libraries(compress
## PUBLIC lz4 snappy zstd zstd_1_3_8 zlib_lite)
compress_cargs = []
if get_option('cpp-standard-20')
    compress_cargs += [ '-Wno-deprecated-non-prototype' ]
endif

lib_compress = static_library(
  'compress',
  sources : compress_sources,
  #include_directories : easy_include_dirs,
  include_directories : oblib_base_without_pass_dirs,
  #dependencies : [ lib_zlib_lite ],
  link_with : [ lib_lz4, lib_snappy, lib_zstd, lib_zstd_1_3_8 ],
  # soversion: soversion,
  # version: libversion,
  c_args : compress_cargs,
  # install: true,
)
