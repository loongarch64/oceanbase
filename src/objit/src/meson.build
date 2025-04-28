objit_base_lib = library('objit_base', 'INTERFACE')
# target_include_directories(['objit_base', 'INTERFACE', cmake_current_source_dir, cmake_current_source_dir, '/../include'])
# GLOB all source files according to its suffix.
# file(['GLOB_RECURSE', 'objit_SRC', '*.h', '*.cpp', '${CMAKE_CURRENT_SROUCE_DIR}/../include/*.h'])
if 'NOT', 'CPP_STANDARD_20'
  # add_definitions('-std=gnu++14')
endif
objit_objects_lib = library('objit_objects', 'OBJECT', objit_src)
# target_link_libraries(['objit_objects', 'PRIVATE', 'oblib_base', 'objit_base'])
# Define a static library target named OBJIT_RAW.
objit_lib = static_library('objit', 'dummy.cpp')
# target_link_libraries(['objit', 'PUBLIC', 'objit_objects', 'objit_base', 'oblib', '${llvm_libs}', 'INTERFACE', '-lrt', '-ldl', '-lpthread', '-lm'])
