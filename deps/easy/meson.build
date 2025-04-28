#project('easy', 'version', '1.1.22', 'languages', 'c', 'cpp', default_options : ['default_library=static'])

# message(['STATUS', 'check deps for libeasy'])
#if 'NOT', 'DEP_DIR'
#  # message(['FATAL_ERROR', 'DEP_DIR not set'])
#endif

#easy_base_lib = library('easy_base', 'INTERFACE')
#inner_include_dirs = [cmake_current_source_dir, '/src/include', cmake_current_source_dir, '/src/']
## set_property(['GLOBAL', 'PROPERTY', 'EASY_INCLUDE_DIRS', inner_include_dirs])

easy_include_dirs = [
  include_directories('./src/include'),
  include_directories('./src')
  ]

easy_test_dirs = include_directories('test')

#if 'OB_USE_BABASSL'
#  # target_include_directories(['easy_base', 'INTERFACE', inner_include_dirs, dep_3rd_dir, '/usr/local/babassl-ob/include'])
#else
#  # target_include_directories(['easy_base', 'INTERFACE', inner_include_dirs, dep_dir, '/include'])
#endif

if get_option('use-babassl')
#  # target_include_directories(['easy_base', 'INTERFACE', inner_include_dirs, dep_3rd_dir, '/usr/local/babassl-ob/include'])
else
#  # target_include_directories(['easy_base', 'INTERFACE', inner_include_dirs, dep_dir, '/include'])
endif

if cc.get_id() == 'clang'
  message('is clang')
else
  message('is not clang')
endif
message(cpp.get_id())

#if 'OB_USE_CLANG'
#  easy_cc_wno = ['-fno-strict-aliasing', '-Wno-unused-variable', '-Wno-unused-function', '-fmax-type-align=8', '-Wno-tautological-constant-out-of-range-compare', '-fno-omit-frame-pointer']
#  if 'CPP_STANDARD_20'
#    easy_cc_wno = [easy_cc_wno, '-Wno-unused-but-set-variable']
#  endif
#else
#  easy_cc_wno = ['-fno-strict-aliasing', '-Wno-unused-variable', '-Wno-implicit-function-declaration', '-Wno-unused-but-set-variable', '-Wno-unused-function', '-fno-omit-frame-pointer']
#endif

# target_compile_options(['easy_base', 'INTERFACE', '-D_GLIBCXX_USE_CXX11_ABI=0', '-D_GNU_SOURCE', '-D__STDC_LIMIT_MACROS', march_cflags, mtune_cflags, '$<$<COMPILE_LANGUAGE:C>:-g', '-Wall', '-Werror', '-fPIC', easy_cc_wno, '>'])
# target_link_libraries(['easy_base', 'INTERFACE', '-L${DEP_DIR}/lib', '-L${DEP_3RD_DIR}/usr/local/babassl-ob/lib/', '-lpthread', '-lm', '-l:libssl.a', '-l:libcrypto.a', arch_ldflags])
# target_compile_features(['easy_base', 'INTERFACE', 'cxx_std_11'])

subdir('src')
#subdir('test')
