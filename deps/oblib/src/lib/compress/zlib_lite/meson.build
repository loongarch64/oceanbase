zlib_lite_cargs = ['-fvisibility=hidden']

if get_option('cpp-standard-20')
  zlib_lite_cargs += ['-Wno-deprecated-non-prototype']
endif

adaptor_cargs = zlib_lite_cargs
qpl_cargs = zlib_lite_cargs

if host_machine.cpu_family() == 'x86_64'
  adaptor_cargs += ['-DENABLE_QPL_COMPRESSION']
  #adaptor_dir = [ depdir + '/include' ]

  qpl_cargs += ['-DENABLE_QPL_COMPRESSION']
  #qpl_dir = [ depdir + '/include' ]

  qpl_deps = []
  # target_link_libraries(['zlib_lite_qpl', 'PUBLIC', '${DEP_DIR}/lib64/libqpl.a'])
  # target_link_libraries(['zlib_lite_qpl', 'PUBLIC', '${DEP_DIR}/lib64/libusdm_drv.a'])
endif

zlib_lite_src_sources = [
    'zlib_lite_src/adler32.c',
    'zlib_lite_src/compress.c',
    'zlib_lite_src/crc32.c',
    'zlib_lite_src/deflate.c',
    'zlib_lite_src/deflate.h',
    'zlib_lite_src/gzclose.c',
    'zlib_lite_src/gzlib.c',
    'zlib_lite_src/gzread.c',
    'zlib_lite_src/gzwrite.c',
    'zlib_lite_src/infback.c',
    'zlib_lite_src/inffast.c',
    'zlib_lite_src/inflate.c',
    'zlib_lite_src/inftrees.c',
    'zlib_lite_src/trees.c',
    'zlib_lite_src/uncompr.c',
    'zlib_lite_src/zutil.c',
]

zlib_lite_adaptor_sources = [
  'zlib_lite_adaptor.cpp'
]

zlib_lite_qpl_sources = [
  'codec_deflate_qpl.cpp'
]

lib_zlib_lite_src = static_library(
  'zlib_lite_src',
  sources : zlib_lite_src_sources,
  include_directories : easy_include_dirs,
  dependencies : easy_deps,
  # soversion: soversion,
  # version: libversion,
  c_args : zlib_lite_cargs,
  # install: true,
)
# target_include_directories(['zlib_lite_src', 'PUBLIC', '${DEP_DIR}/include/'])

lib_zlib_lite_adaptor = static_library(
  'zlib_lite_adaptor',
  sources : zlib_lite_adaptor_sources,
  c_args : adaptor_cargs,
)
# target_compile_options(['zlib_lite_adaptor', 'PRIVATE', zlib_lite_compile_definitions])

# We can not put the file codec_deflate_qpl.cpp into zlib_lite_objs.
# I got some link error messages like below:
# ld.lld: error: relocation refers to a symbol in a discarded section: __clang_call_terminate
#zlib_lite_qpl_lib = library('zlib_lite_qpl', 'codec_deflate_qpl.cpp')
lib_zlib_lite_qpl = static_library(
  'zlib_lite_qpl',
  sources : zlib_lite_qpl_sources,
  c_args : qpl_cargs,
)

#add_custom_command(['OUTPUT', 'zlib_lite_objs.o',
#'COMMAND', ob_ld_bin,
#'-r',
#'$<TARGET_OBJECTS:zlib_lite_src>',
#'$<TARGET_OBJECTS:zlib_lite_adaptor>',
#'-o',
#'zlib_lite_objs.tmp.o',
#
#'COMMAND', ob_objcopy_bin,
#'--localize-hidden',
#'zlib_lite_objs.tmp.o',
#'zlib_lite_objs.o',
#'DEPENDS', 'zlib_lite_src', 'zlib_lite_adaptor',
#'BYPRODUCTS', 'zlib_lite_objs.o',
#'COMMAND_EXPAND_LISTS'])

#ld_bin = find_program('ld')
#objcopy_bin = find_program('objcopy')
#
#zlib_lite_objs_tmp_o = custom_target(
#  'zlib_lite_objs.tmp.o',
#  output : 'zlib_lite_objs.tmp.o',
#  depends : [lib_zlib_lite_src, lib_zlib_lite_adaptor],
#  command : [ld_bin, #cc.get_linker_id(),
#    '-r',
#    lib_zlib_lite_src,
#    lib_zlib_lite_adaptor,
#    '-o',
#    '@OUTPUT@'])
#
#zlib_lite_objs_o = custom_target(
#  'zlib_lite_objs.o',
#  input : zlib_lite_objs_tmp_o,
#  output : 'zlib_lite_objs.o',
#  command : [objcopy_bin,
#    '--localize-hidden',
#    '@INPUT@',
#    '@OUTPUT@'])
#
#lib_zlib_lite = static_library(
#  'zlib_lite',
#  zlib_lite_objs_o,
#  link_with: lib_zlib_lite_qpl,
#)
## oblib_add_extra_objects([cmake_current_binary_dir, '/zlib_lite_objs.o'])
## oblib_add_library(['zlib_lite', 'zlib_lite_objs.o'])
## target_link_libraries(['zlib_lite', 'zlib_lite_qpl'])

lib_zlib_lite = static_library(
  'zlib_lite',
  sources : [
    zlib_lite_src_sources,
    zlib_lite_adaptor_sources,
    zlib_lite_qpl_sources,
  ],
  include_directories : easy_include_dirs,
  dependencies : easy_deps,
  # soversion: soversion,
  # version: libversion,
  c_args : zlib_lite_cargs,
  # install: true,
)

## function(['oblib_add_library', 'target'])
## message(['STATUS', 'oblib_add_library ${target}'])
# target_lib = library(target, 'OBJECT', argn)
## list(['APPEND', 'oblib_object_libraries', target])
#oblib_object_libraries = ['${oblib_object_libraries}', 'CACHE', 'INTERNAL', 'oblib_object_libraries']
## endfunction()

## function('oblib_add_extra_objects')
## list(['APPEND', 'oblib_extra_objects', argn])
#oblib_extra_objects = ['${oblib_extra_objects}', 'CACHE', 'INTERNAL', 'oblib_extra_objects']
## endfunction()
