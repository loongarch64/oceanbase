# easy_test_root_dir = include_directories('.')
#
# # function(['easy_test', 'case'])
# # get_property(['EASY_INCLUDE_DIRS', 'GLOBAL', 'PROPERTY', 'EASY_INCLUDE_DIRS'])
# case_lib = library(case, '_base', 'OBJECT', argn)
# # target_include_directories([case, '_base', 'PUBLIC', easy_test_root])
# # target_link_libraries([case, '_base', 'PUBLIC', 'easy_base'])
# # target_compile_features([case, '_base', 'PUBLIC', 'cxx_std_11'])
# case_exe = executable(case, case, '.c')
# # target_link_libraries([case, 'PRIVATE', case, '_base', 'pthread', 'm', 'dl', 'easy'])
# test(case, case)
# # endfunction()

# easy_test_dirs = include_directories('.')

test_sources = [
  #include
  'include/easy_atomic_test.c',
  'include/easy_list_test.c',
  # io
  'io/easy_baseth_pool_test.c',
  'io/easy_client_test.c',
  'io/easy_connection_test.c',
  'io/easy_file_test.c',
  'io/easy_io_test.c',
  'io/easy_log_test.c',
  'io/easy_request_test.c',
  # subdir('memory')
  'memory/easy_mem_page_test.c',
  'memory/easy_mem_pool_test.c',
  'memory/easy_mem_slab_test.c',
  # subdir('packet')
  'packet/easy_http_handler_test.c',
  # subdir('thread')
  'thread/easy_uthread_test.c',
  # subdir('util')
  'util/easy_buf_test.c',
  'util/easy_hash_test.c',
  'util/easy_inet_test.c',
  'util/easy_list_test.c',
  'util/easy_pool_test.c',
  'util/easy_string_test.c',
  'util/easy_time_test.c',
]

foreach src : test_sources
  src_array = src.split('.')
  name = src_array[0].replace('/', '_')
  test(name,
    executable(name,
      sources: [src],
      c_args : [ '-save-temps', easy_base_c_args, '-mno-direct-extern-access'],
      include_directories : [
	include_directories('.'),
	include_directories('..'),
	easy_include_dirs,
	],
      #dependencies : libeasy_dep,
      link_with : [ libeasy ],
      #install : false,
    )
  )
endforeach
