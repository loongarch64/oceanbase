cos_wrapper_sources = [
  'ob_singleton.h',
  'ob_cos_wrapper.h',
  'ob_cos_wrapper.cpp',
  ]

cos_wrapper_cargs = []
cos_wrapper_cppargs = []

lib_cos_wrapper = static_library(
  'cos_wrapper',
  sources : cos_wrapper_sources,
  #include_directories : easy_include_dirs,
  include_directories : oblib_base_without_pass_dirs,
  dependencies : [
    apr_dep,
    cos_c_sdk_dep,
    #mxml_dep,
    ],
  #link_with : [ lib_lz4, lib_snappy, lib_zstd, lib_zstd_1_3_8 ],
  # soversion: soversion,
  # version: libversion,
  c_args : cos_wrapper_cargs,
  # install: true,
  )

#target_link_libraries(cos_wrapper PUBLIC oblib_base)
#
## we need to get all .o files from cos-c-sdk rpm and mixed them into one .o file
#add_custom_command(
#  OUTPUT cos_sdk_lib.o
#  COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/cos_a2o && \
    #      mkdir ${CMAKE_CURRENT_BINARY_DIR}/cos_a2o && \
    #      cp ${DEP_DIR}/lib/libcos_c_sdk_static.a ${CMAKE_CURRENT_BINARY_DIR}/cos_a2o && \
    #      cd ${CMAKE_CURRENT_BINARY_DIR}/cos_a2o && \
    #      ar x libcos_c_sdk_static.a && \
    #      ${OB_LD_BIN} -r cos_*.o -o cos_sdk_lib.o
#  DEPENDS ${DEP_DIR}/lib/libcos_c_sdk_static.a
#)
#
## we need to join cos-c-sdk .o file with cos_wrapper .o file
## cuz we need to set 'localize-hidden', which can help cos_wrapper to get hidden symbol of cos-c-sdk.
#add_custom_command(
#  OUTPUT cos_sdk_objs.o
#  COMMAND ${OB_LD_BIN} -r $<TARGET_OBJECTS:cos_wrapper> -o cos_wrapper.o
#  COMMAND ${OB_LD_BIN} -r cos_wrapper.o ${CMAKE_CURRENT_BINARY_DIR}/cos_a2o/cos_sdk_lib.o -o cos_sdk_objs.tmp.o
#  COMMAND ${OB_OBJCOPY_BIN} --localize-hidden cos_sdk_objs.tmp.o cos_sdk_objs.o
#  DEPENDS cos_wrapper cos_sdk_lib.o
#  BYPRODUCTS cos_sdk_objs.o
#  COMMAND_EXPAND_LISTS
#)
#
#oblib_add_extra_objects(${CMAKE_CURRENT_BINARY_DIR}/cos_sdk_objs.o)
#oblib_add_library(cos_sdk cos_sdk_objs.o)
