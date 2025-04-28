project('objit', 'version', '1.0.0', 'languages', 'c', 'cpp', default_options : ['default_library=static'])
# message(['STATUS', '${DEP_3RD_DIR}'])
if 'NOT', 'DEP_3RD_DIR'
  # message(['FATAL_ERROR', 'DEP_3RD_DIR not set'])
endif
llvm_dir = '${DEP_3RD_DIR}/usr/local/oceanbase/devtools/lib/cmake/llvm'
LLVM_dep = dependency('LLVM')
# message(['STATUS', 'This is BINARY dir ', project_binary_dir])
# message(['STATUS', 'This is SOURCE dir ', project_source_dir])
# include_directories(llvm_include_dirs)
# add_definitions(llvm_definitions)
if 'NOT', 'CPP_STANDARD_20'
  # add_definitions('-D_GLIBCXX_USE_CXX11_ABI=0')
endif
# add_definitions(['-g', '-O2', '-frtti'])
# Find the libraries that correspond to the LLVM components
# that we wish to use
if architecture, 'STREQUAL', 'x86_64'
  # llvm_map_components_to_libnames(['llvm_libs', 'Support', 'Core', 'IRReader', 'ExecutionEngine', 'OrcJit', 'McJit', 'X86CodeGen', 'X86AsmParser', 'runtimedyld', 'bitreader', 'bitwriter', 'object', 'objectyaml', 'target', 'DebugInfoDWARF', 'Symbolize'])
else
  # llvm_map_components_to_libnames(['llvm_libs', 'Support', 'Core', 'IRReader', 'ExecutionEngine', 'OrcJit', 'McJit', 'AArch64CodeGen', 'AArch64AsmParser', 'runtimedyld', 'bitreader', 'bitwriter', 'object', 'objectyaml', 'target', 'DebugInfoDWARF', 'Symbolize'])
endif
cmake_module_path = '${LLVM_CMAKE_DIR}'
subdir('src')
