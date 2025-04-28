#  # easy_test(['include_test', 'easy_atomic_test.c', 'easy_list_test.c'])
#
#  easy_test_root = cmake_current_source_dir
#  # function(['easy_test', 'case'])
#  # get_property(['EASY_INCLUDE_DIRS', 'GLOBAL', 'PROPERTY', 'EASY_INCLUDE_DIRS'])
#  case_lib = library(case, '_base', 'OBJECT', argn)
#  # target_include_directories([case, '_base', 'PUBLIC', easy_test_root])
#  # target_link_libraries([case, '_base', 'PUBLIC', 'easy_base'])
#  # target_compile_features([case, '_base', 'PUBLIC', 'cxx_std_11'])
#  case_exe = executable(case, case, '.c')
#  # target_link_libraries([case, 'PRIVATE', case, '_base', 'pthread', 'm', 'dl', 'easy'])
#  test(case, case)
#  # endfunction()

include_sources = [
  'easy_atomic_test.c',
  'easy_list_test.c',
]

foreach src : include_sources
  src_array = src.split('.')
  test(src_array.get(0),
    executable(src_array.get(0),
      src,
      install : false,
      c_args : cargs,
      include_directories : [
	easy_include_path,
	include_directories('..'),
	],
      dependencies : easy_deps,
      link_with : [
	libeasy
	],
    )
  )
endforeach
