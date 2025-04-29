# # function(['easy_add_objects', 'target'])
# # foreach(['item', argn])
# if 'item', 'MATCHES', '.*\.h$'
#   # list(['APPEND', 'easy_headers', '${target}/${item}'])
# else
#   # list(['APPEND', 'easy_objects', '${target}/${item}'])
# endif
# # endforeach('item')
# easy_headers = ['${easy_headers}', 'CACHE', 'INTERNAL', 'easy_headers']
# easy_objects = ['${easy_objects}', 'CACHE', 'INTERNAL', 'easy_objects']
# # endfunction()
# easy_headers = ['', 'CACHE', 'INTERNAL', 'easy_headers']
# easy_objects = ['', 'CACHE', 'INTERNAL', 'easy_objects']

#subdir('io')
io_sources = ([
  'io/easy_baseth_pool.c',
  'io/easy_client.c',
  'io/easy_connection.c',
  'io/easy_file.c',
  'io/easy_io.c',
  'io/easy_log.c',
  'io/easy_maccept.c',
  'io/easy_message.c',
  'io/easy_request.c',
  'io/easy_socket.c',
  'io/easy_ssl.c',
  'io/easy_summary.c',
  'io/easy_negotiation.c',
  'io/ev.c']
)

#subdir('memory')
memory_sources = [
  'memory/easy_mem_page.c',
  'memory/easy_mem_pool.c',
  'memory/easy_mem_slab.c'
]

#subdir('packet')
packet_sources = [
  'packet/http/easy_http_handler.c',
  'packet/http/http_parser.c'
]

#subdir('thread')
thread_sources = ['thread/easy_uthread.c']

#subdir('util')
util_sources = [
  'util/easy_array.c',
  'util/easy_buf.c',
  'util/easy_hash.c',
  'util/easy_inet.c',
  'util/easy_mod_stat.c',
  'util/easy_pool.c',
  'util/easy_string.c',
  'util/easy_time.c',
  'util/easy_util.c'
]

if cc.get_id() == 'clang'
  message('is clang')
  cargs = [
  '-fno-strict-aliasing',
  '-Wno-unused-variable',
  '-Wno-unused-function',
  '-fmax-type-align=8',
  '-Wno-tautological-constant-out-of-range-compare',
  '-fno-omit-frame-pointer' ]
#  if 'CPP_STANDARD_20'
#    easy_cc_wno = [easy_cc_wno, '-Wno-unused-but-set-variable']
#  endif
else
  message('is not clang')
  cargs = [
    '-fno-strict-aliasing',
    '-Wno-unused-variable',
    '-Wno-implicit-function-declaration',
    '-Wno-unused-but-set-variable',
    '-Wno-unused-function',
    '-fno-omit-frame-pointer']
endif

# target_compile_options(['easy_base', 'INTERFACE', '-D_GLIBCXX_USE_CXX11_ABI=0', '-D_GNU_SOURCE', '-D__STDC_LIMIT_MACROS', march_cflags, mtune_cflags, '$<$<COMPILE_LANGUAGE:C>:-g', '-Wall', '-Werror', '-fPIC', easy_cc_wno, '>'])
# target_link_libraries(['easy_base', 'INTERFACE', '-L${DEP_DIR}/lib', '-L${DEP_3RD_DIR}/usr/local/babassl-ob/lib/', '-lpthread', '-lm', '-l:libssl.a', '-l:libcrypto.a', arch_ldflags])
# target_compile_features(['easy_base', 'INTERFACE', 'cxx_std_11'])
cargs += [
'-D_GLIBCXX_USE_CXX11_ABI=0',
'-D_GNU_SOURCE',
'-D__STDC_LIMIT_MACROS',
'-D_NO_EXCEPTION',
#'-g',
#'-Wall',
#'-Werror',
'-fPIC',
]

easy_include_dirs = [include_directories('include'), include_directories('.')]

libeasy = static_library(
  'easy',
  sources : [
    io_sources,
    memory_sources,
    packet_sources,
    thread_sources,
    util_sources
  ],
  include_directories : easy_include_dirs,
  dependencies : easy_deps,
  # soversion: soversion,
  # version: libversion,
  c_args : cargs,
  # install: true,
)
