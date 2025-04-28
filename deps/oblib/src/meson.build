#oblib_base_lib = library('oblib_base',
#'INTERFACE')
#
#oblib_base_without_pass_lib = library('oblib_base_without_pass',
#'INTERFACE')
#
#oblib_pass_lib = library('oblib_pass',
#'INTERFACE')
#
#if 'NOT', 'CPP_STANDARD_20'
#  # target_compile_options(['oblib_base_without_pass', 'INTERFACE', '-D_GLIBCXX_USE_CXX11_ABI=0'])
#endif
#
### get_property(['EASY_INCLUDE_DIRS', 'GLOBAL', 'PROPERTY', 'EASY_INCLUDE_DIRS'])

oblib_base_without_pass_dirs = [
  include_directories('.'),
  include_directories('..'),
  include_directories('../../..'),
  include_directories('../../../src'),
  include_directories('../../../src/plugin/include'),
  include_directories('../../../deps/easy/src'),
  include_directories('../../../deps/oblib/src'),
  include_directories('../../../deps/oblib/src/common'),
  include_directories('../../../deps/easy/src/include'),
  include_directories('../../../deps/ussl-hook'),
  include_directories('../../../deps/ussl-hook/loop'),
  include_directories('../../../src/objit/include'),
  include_directories('../../../src/objit/src'),
]

## target_include_directories(['oblib_base_without_pass',
#'INTERFACE',
#cmake_source_dir,
#cmake_source_dir, '/src',
#cmake_source_dir, '/src/plugin/include',
#cmake_source_dir, '/deps/easy/src',
#cmake_source_dir, '/deps/oblib/src',
#cmake_source_dir, '/deps/oblib/src/common',
#cmake_source_dir, '/deps/easy/src/include',
#cmake_source_dir, '/deps/ussl-hook',
#cmake_source_dir, '/deps/ussl-hook/loop',
#cmake_source_dir, '/src/objit/include',
#cmake_source_dir, '/src/objit/src',
#dep_dir, '/include',
#dep_dir, '/include/libxml2',
#devtools_dir,
#devtools_dir, '/include',
#dep_dir, '/include/apr-1/',
#dep_dir, '/include/icu',
#dep_dir, '/include/icu/common',
#dep_dir, '/include/apache-arrow',
#dep_dir, '/include/apache-orc',
#dep_dir, '/include/fast_float',
#ussl_include_dirs])
#
#if 'OB_BUILD_OPENSOURCE'
#  # target_include_directories(['oblib_base_without_pass', 'INTERFACE', dep_dir, '/include/mariadb'])
#endif
#if 'OB_BUILD_SPM'
#  # target_include_directories(['oblib_base_without_pass', 'INTERFACE', cmake_source_dir, '/close_modules/spm'])
#endif
#if 'OB_BUILD_DBLINK'
#  # target_include_directories(['oblib_base_without_pass', 'INTERFACE', dep_3rd_dir, '/u01/obclient/include', cmake_source_dir, '/close_modules/dblink/deps/oblib/src', oci_devel_inc])
#endif
#if 'OB_BUILD_ARBITRATION'
#  # target_include_directories(['oblib_base_without_pass', 'INTERFACE', cmake_source_dir, '/close_modules/arbitration'])
#endif
#if 'OB_BUILD_SHARED_STORAGE'
#  # target_include_directories(['oblib_base_without_pass', 'INTERFACE', cmake_source_dir, '/close_modules/shared_storage/'])
#endif
#if 'OB_BUILD_LOG_STORAGE_COMPRESS'
#  # target_include_directories(['oblib_base_without_pass', 'INTERFACE', cmake_source_dir, '/close_modules/log_storage_compress'])
#endif
#if 'OB_BUILD_ORACLE_PL'
#  # target_include_directories(['oblib_base_without_pass', 'INTERFACE', cmake_source_dir, '/close_modules/oracle_pl'])
#endif
#if 'OB_BUILD_TDE_SECURITY'
#  # target_include_directories(['oblib_base_without_pass', 'INTERFACE', cmake_source_dir, '/close_modules/tde_security'])
#endif
#if 'OB_BUILD_AUDIT_SECURITY'
#  # target_include_directories(['oblib_base_without_pass', 'INTERFACE', cmake_source_dir, '/close_modules/audit_security'])
#endif
#if 'OB_USE_BABASSL'
#  # target_include_directories(['oblib_base_without_pass', 'INTERFACE', dep_3rd_dir, '/usr/local/babassl-ob/include'])
#endif
#if 'OB_USE_CLANG'
#  # The following clang warnings should be fixed later: -Wno-unused-variable -Wno-invalid-offsetof
#  oblib_compile_definitions = ['-fno-strict-aliasing', '-fno-omit-frame-pointer', march_cflags, mtune_cflags, '-D__STDC_LIMIT_MACROS', '-D__STDC_CONSTANT_MACROS', '-D_NO_EXCEPTION', '-DOCI_LINK_RUNTIME', '-Wall', '-Wextra', '-Wformat', '-Werror', '-Wno-deprecated', '-Wno-address-of-packed-member', '-Wno-sign-compare', '-Wno-tautological-compare', '-Wno-psabi', '-Wno-c99-designator', '-Wno-int-in-bool-context', '-Wno-sizeof-array-div', '-Wno-implicit-const-int-float-conversion', '-Wno-fortify-source', '-Wno-non-c-typedef-for-linkage', '-Wno-uninitialized-const-reference', '-Wno-unused-function', '-Wno-atomic-alignment', '-Wno-tautological-constant-out-of-range-compare', '-Wno-unused-parameter', '-Wno-string-plus-int', '-Wno-misleading-indentation', '-Wno-unused-private-field', '$<$<COMPILE_LANGUAGE:CXX>:-Wno-overloaded-virtual', '-Wno-reserved-user-defined-literal', '-Wno-self-assign', '-Wno-inconsistent-missing-override', '-Wno-mismatched-tags', '-Wno-unused-variable', '-Wno-invalid-offsetof', '-Wno-unevaluated-expression', '-Wno-constant-logical-operand>', '$<$<COMPILE_LANGUAGE:C>:-Wno-unused-parameter>']
#  if 'CPP_STANDARD_20'
#    oblib_compile_definitions = [oblib_compile_definitions, '-Wno-unused-but-set-variable', '-Wno-null-pointer-subtraction', '-Wno-compound-token-split-by-macro', '-Wno-packed-non-pod', '-Wno-format', '-Wno-shorten-64-to-32', '-Wno-unused-but-set-parameter', '-Wno-single-bit-bitfield-constant-conversion', '-Wno-enum-constexpr-conversion', '-Wno-covered-switch-default', '-Wno-bitwise-instead-of-logical', '-Wno-suggest-override', '-Wno-non-virtual-dtor', '-Wno-cast-qual', '-Wno-unused-value', '-Wno-macro-redefined', '-Wno-reinterpret-base-class', '-Wno-register', '-Wno-implicit-fallthrough', '-Wno-deprecated-non-prototype', '-Wno-ambiguous-reversed-operator', '-Wno-invalid-partial-specialization', '-Wno-string-concatenation']
#  endif
#  # target_compile_options(['oblib_base_without_pass', 'INTERFACE', oblib_compile_definitions])
#  if 'ENABLE_SANITY'
#    # target_compile_options(['oblib_pass', 'INTERFACE', '$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xclang -load -Xclang ${DEVTOOLS_DIR}/lib64/libsanitypass.so>', '-fno-builtin-memset', '-fno-builtin-bzero', '-fno-builtin-memcpy', '-fno-builtin-memmove', '-fno-builtin-memcmp', '-fno-builtin-strlen', '-fno-builtin-strnlen', '-fno-builtin-strcpy', '-fno-builtin-strncpy', '-fno-builtin-strcmp', '-fno-builtin-strncmp', '-fno-builtin-strcasecmp', '-fno-builtin-strncasecmp', '-fno-builtin-vsprintf', '-fno-builtin-vsnprintf', '-fno-builtin-sprintf', '-fno-builtin-snprintf'])
#  endif
#else
#  # target_compile_options(['oblib_base_without_pass', 'INTERFACE', '-fno-strict-aliasing', '-fno-omit-frame-pointer', march_cflags, mtune_cflags, '-D__STDC_LIMIT_MACROS', '-D__STDC_CONSTANT_MACROS', '-D_NO_EXCEPTION', '-DOCI_LINK_RUNTIME', '-Wall', '-Wextra', '-Wformat', '-Werror', '-Wno-deprecated', '-Wno-sign-compare', '-Wno-psabi', '-Wno-attributes', '-Wno-nonnull-compare', '-Wno-parentheses', '-Wno-implicit-fallthrough', '-Wno-switch', '-Wno-address-of-packed-member', '-Wno-ignored-qualifiers', '-Wno-format', '-Wno-type-limits', '-Wno-format-truncation', '-Wno-int-in-bool-context', '-Wno-cast-function-type', '-Wno-unused-function', '-Wno-overflow', '-Wno-unused-but-set-variable', '-Wno-memset-elt-size', '$<$<COMPILE_LANGUAGE:CXX>:', '-Wno-literal-suffix', '-Wno-invalid-offsetof', '-Wno-unused-variable', '-Wno-deprecated-copy', '-Wno-class-memaccess', '-Wno-unused-parameter', '-Wno-unused-but-set-variable', '-Wno-pragmas', '-Wno-maybe-uninitialized', '-Wno-subobject-linkage>', '$<$<COMPILE_LANGUAGE:C>:', '-Wno-unused-parameter>'])
#endif
## target_compile_features(['oblib_base', 'INTERFACE', 'cxx_std_11'])
#oblib_abseil_list = [dep_dir, '/lib64/libabsl_base.a', dep_dir, '/lib64/libabsl_city.a', dep_dir, '/lib64/libabsl_debugging_internal.a', dep_dir, '/lib64/libabsl_demangle_internal.a', dep_dir, '/lib64/libabsl_hash.a', dep_dir, '/lib64/libabsl_int128.a', dep_dir, '/lib64/libabsl_low_level_hash.a', dep_dir, '/lib64/libabsl_malloc_internal.a', dep_dir, '/lib64/libabsl_raw_hash_set.a', dep_dir, '/lib64/libabsl_raw_logging_internal.a', dep_dir, '/lib64/libabsl_spinlock_wait.a', dep_dir, '/lib64/libabsl_stacktrace.a', dep_dir, '/lib64/libabsl_str_format_internal.a', dep_dir, '/lib64/libabsl_strings.a', dep_dir, '/lib64/libabsl_strings_internal.a', dep_dir, '/lib64/libabsl_symbolize.a', dep_dir, '/lib64/libabsl_synchronization.a', dep_dir, '/lib64/libabsl_throw_delegate.a', dep_dir, '/lib64/libabsl_time.a', dep_dir, '/lib64/libabsl_time_zone.a']
#if 'OB_BUILD_OPENSOURCE'
#  lgpl_deps = '-L${DEP_DIR}/lib/mariadb -lmariadb'
#  if 'OB_STATIC_LINK_LGPL_DEPS'
#    lgpl_deps = '-L${DEP_DIR}/lib/mariadb -l:libmariadbclient.a'
#  endif
#  # target_link_libraries(['oblib_base_without_pass', 'INTERFACE', 'oss', 'easy', dep_dir, '/lib/libcurl.a', dep_dir, '/lib/libssl.a', dep_dir, '/lib/libcrypto.a', dep_dir, '/lib64/libs2.a', oblib_abseil_list, dep_dir, '/lib/libroaring.a', lgpl_deps, '$<$<STREQUAL:', '${ARCHITECTURE}', ',', 'x86_64', '>:${DEP_DIR}/lib/libunwind.a>', dep_dir, '/lib/libz.a', dep_dir, '/lib/libicui18n.a', dep_dir, '/lib/libicustubdata.a', dep_dir, '/lib/libicuuc.a', dep_dir, '/lib/libprotobuf-c.a', dep_dir, '/lib64/libarrow.a', dep_dir, '/lib64/libparquet.a', dep_dir, '/lib64/libarrow_bundled_dependencies.a', dep_dir, '/lib64/liborc.a', dep_dir, '/lib64/libsnappy.a', dep_dir, '/lib64/libprotoc.a', dep_dir, '/lib64/libprotobuf.a', dep_dir, '/lib64/liblz4.a', dep_dir, '/lib64/libzstd.a', '-L${DEP_DIR}/var/usr/lib64', '-L${DEP_DIR}/var/usr/lib', '-L${DEP_3RD_DIR}/usr/lib', '-L${DEP_3RD_DIR}/usr/lib64', '-laio', '-lpthread', '-ldl', '-lrt', arch_ldflags, dep_dir, '/lib/libisal.a'])
#else
#  ignoreme = 'OB_STATIC_LINK_LGPL_DEPS'
#  # target_link_libraries(['oblib_base_without_pass', 'INTERFACE', 'oss', 'easy', '$<$<STREQUAL:', '${ARCHITECTURE}', ',', 'x86_64', '>:${DEP_DIR}/lib/libunwind.a>', dep_dir, '/lib/libcurl.a', dep_3rd_dir, '/usr/local/babassl-ob/lib/libssl.a', dep_3rd_dir, '/usr/local/babassl-ob/lib/libcrypto.a', dep_3rd_dir, '/u01/obclient/lib/libobclnt.a', dep_dir, '/lib64/libs2.a', oblib_abseil_list, dep_dir, '/lib/libroaring.a', dep_dir, '/lib/libz.a', dep_dir, '/lib/libicui18n.a', dep_dir, '/lib/libicustubdata.a', dep_dir, '/lib/libicuuc.a', dep_dir, '/lib/libprotobuf-c.a', dep_dir, '/lib64/libarrow.a', dep_dir, '/lib64/libparquet.a', dep_dir, '/lib64/libarrow_bundled_dependencies.a', dep_dir, '/lib64/liborc.a', dep_dir, '/lib64/libsnappy.a', dep_dir, '/lib64/libprotoc.a', dep_dir, '/lib64/libprotobuf.a', dep_dir, '/lib64/liblz4.a', dep_dir, '/lib64/libzstd.a', '-L${DEP_DIR}/var/usr/lib64', '-L${DEP_DIR}/var/usr/lib', '-L${DEP_3RD_DIR}/usr/lib', '-L${DEP_3RD_DIR}/usr/lib64', '-laio', '-lpthread', '-ldl', '-lrt', arch_ldflags, dep_dir, '/lib/libisal.a'])
#endif
## target_link_libraries(['oblib_base', 'INTERFACE', 'oblib_base_without_pass', 'oblib_pass'])
#
#
#
## function(['oblib_add_library', 'target'])
## message(['STATUS', 'oblib_add_library ${target}'])
#target_lib = library(target, 'OBJECT', argn)
## list(['APPEND', 'oblib_object_libraries', target])
#oblib_object_libraries = ['${oblib_object_libraries}', 'CACHE', 'INTERNAL', 'oblib_object_libraries']
## endfunction()

## function('oblib_add_extra_objects')
## list(['APPEND', 'oblib_extra_objects', argn])
#oblib_extra_objects = ['${oblib_extra_objects}', 'CACHE', 'INTERNAL', 'oblib_extra_objects']
## endfunction()
#
#oblib_object_libraries = ['', 'CACHE', 'INTERNAL', 'oblib_object_libraries']
#oblib_extra_objects = ['', 'CACHE', 'INTERNAL', 'oblib_extra_objects']

subdir('lib')
#subdir('common')
#subdir('rpc')
