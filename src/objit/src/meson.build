args = [ '-g -O2 -frtti']
objit_objects_sources = [
  'core/jit_context.cpp',
  'core/ob_jit_allocator.cpp',
  'core/ob_jit_memory_manager.cpp',
  'core/ob_orc_jit.cpp',
  'core/ob_pl_ir_compiler.cpp',
  'expr/ob_case_op_expr.cpp',
  'expr/ob_column_ref_expr.cpp',
  'expr/ob_const_expr.cpp',
  'expr/ob_iraw_expr.cpp',
  'expr/ob_op_expr.cpp',
  'expr/ob_var_expr.cpp',
  'expr/string_cmp.cpp',
  'expr/ob_llvm_type.cpp',
  'dummy.cpp',
  'ob_llvm_di_helper.cpp',
  'ob_llvm_helper.cpp',
  'ob_llvm_symbolizer.cpp',
  ]
lib_objit_objects = library(
  'objit_objects',
  sources : objit_objects_sources,
  include_directories: [
    include_directories('../include' )
  ],
  if get_option('CPP_STANDARD_20')
    cpp_args += ['-std=gnu++14','-D_GLIBCXX_USE_CXX11_ABI=0'],
  endif
  link_with: oblib_base,
  )
lib_objit_lib = static_library(
  'objit',
  sources: 'dummy.cpp',
  include_directories: [
    include_directories('../include' )
  ],
  if get_option('CPP_STANDARD_20')
    cpp_args += ['-std=gnu++14','-D_GLIBCXX_USE_CXX11_ABI=0'],
  endif
  link_with: [lib_oblib_base, lib_objit_objects, lib_oblib],
  dependencies: [llvm_dep, librt, libdl, libpthread, libm],
  )
