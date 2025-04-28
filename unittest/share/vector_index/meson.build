if 'OB_BUILD_OPENSOURCE'
  # ob_unittest('test_vector_index_serialize')
else
  data_archive = '${CMAKE_CURRENT_SOURCE_DIR}/data/dataset.tar.gz'
  build_data_dir = '${CMAKE_CURRENT_BINARY_DIR}/data'
  # file(['MAKE_DIRECTORY', build_data_dir])
  # execute_process(['COMMAND', 'sh', '-c', 'tar xzf ${DATA_ARCHIVE} --directory ${BUILD_DATA_DIR}', 'RESULT_VARIABLE', 'RET'])
  if 'RET', 'AND', 'NOT', 'RET', 'EQUAL', '0'
    # message(['FATAL_ERROR', 'failed to extract dataset in ${DATA_ARCHIVE}'])
  endif
  # ob_unittest('test_vector_index_serialize')
  # ob_unittest('test_kmeans')
  # ob_unittest('test_simd_vector_op')
endif
