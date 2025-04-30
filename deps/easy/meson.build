easy_include_dirs = [
  include_directories('./src/include'),
  include_directories('./src'),
  ]

if get_option('use-babassl')
#  # target_include_directories(['easy_base', 'INTERFACE', inner_include_dirs, dep_3rd_dir, '/usr/local/babassl-ob/include'])
else
#  # target_include_directories(['easy_base', 'INTERFACE', inner_include_dirs, dep_dir, '/include'])
endif

if cc.get_id() == 'clang'
  easy_cc_wno = [
    '-fno-strict-aliasing',
    '-Wno-unused-variable',
    '-Wno-unused-function',
    '-fmax-type-align=8',
    '-Wno-tautological-constant-out-of-range-compare',
    '-fno-omit-frame-pointer']
  if get_option('cpp-standard-20')
    easy_cc_wno += [ '-Wno-unused-but-set-variable']
  endif
else
  easy_cc_wno = [
    '-fno-strict-aliasing',
    '-Wno-unused-variable',
    '-Wno-implicit-function-declaration',
    '-Wno-unused-but-set-variable',
    '-Wno-unused-function',
    '-fno-omit-frame-pointer']
endif

easy_base_c_args = [
  '-D_GLIBCXX_USE_CXX11_ABI=0',
  '-D_GNU_SOURCE',
  '-D__STDC_LIMIT_MACROS',
  #march_cflags,
  #mtune_cflags,
  '-g',
  #'-Wall',
  #'-Werror',
  #  '-fPIC',
  easy_cc_wno,
  ]

easy_base_cpp_args = [
  '-D_GLIBCXX_USE_CXX11_ABI=0',
  '-D_GNU_SOURCE',
  '-D__STDC_LIMIT_MACROS',
  #march_cflags,
  #mtune_cflags,
  ]

# target_link_libraries(['easy_base', 'INTERFACE', '-L${DEP_DIR}/lib', '-L${DEP_3RD_DIR}/usr/local/babassl-ob/lib/', '-lpthread', '-lm', '-l:libssl.a', '-l:libcrypto.a', arch_ldflags])
# target_compile_features(['easy_base', 'INTERFACE', 'cxx_std_11'])
easy_deps = [
  openssl_dep,
]

libm_dep = cc.find_library('m', required: false)
if libm_dep.found()
  easy_deps += [libm_dep]
endif

libpthread_dep = cc.find_library('pthread', required: false)
if libpthread_dep.found()
  easy_deps += [libpthread_dep]
endif

subdir('src')

# FIXME: test failed
# add_subdirectory(test EXCLUDE_FROM_ALL)
#subdir('test')
