util_file_name = 'ob_log_miner_test_utils.cpp'
# function(['logminer_unittest', 'case'])
if 'ARGC', 'EQUAL', '1'
  case_exe = executable(case, case, '.cpp', util_file_name)
else
  ARGV_exe = executable(argv, util_file_name)
endif
if 'case', 'MATCHES', '^test_.*'
  test(case, case)
  # set_tests_properties([case, 'PROPERTIES', 'TIMEOUT', '300'])
endif
# target_link_libraries([case, 'PRIVATE', 'oblogminer_obj_dev', 'gtest', 'gmock'])
# target_include_directories([case, 'PRIVATE', cmake_current_source_dir, cmake_source_dir, '/unittest', cmake_source_dir, '/deps/oblib/unittest'])
# endfunction()
# logminer_unittest('test_ob_log_miner_args')
# logminer_unittest('test_ob_log_miner_br_filter')
# logminer_unittest('test_ob_log_miner_record')
# logminer_unittest('test_ob_log_miner_record_converter')
# logminer_unittest('test_ob_log_miner_utils')
# logminer_unittest('test_ob_log_miner_analyzer_checkpoint')
# logminer_unittest('test_ob_log_miner_progress_range')
# logminer_unittest('test_ob_log_miner_file_meta')
# logminer_unittest('test_ob_log_miner_file_index')
# logminer_unittest('test_ob_log_miner_file_manager')
