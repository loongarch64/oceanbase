ob_link_args = [
  '-Wl,-z,relro,-z,now'
]

ob_c_args = [
  '-DPACKAGE_NAME="@0@"'.format(meson.project_name()),
  '-DPACKAGE_VERSION="@0@"'.format(meson.project_version()),
  '-DPACKAGE_STRING="@0@-@1@"'.format(meson.project_name(), meson.project_version()),
  # change to real release ID.
  '-DRELEASEID="1"'
 ]

ob_c_args += '-Wno-invalid-offsetof'
oblib_pass_c_args = [
      '-Xclang',
      '-load',
      '-Xclang',
      ${DEVTOOLS_DIR}/lib64/libsanitypass.so,
      '-fno-builtin-memset',
      '-fno-builtin-bzero',
      '-fno-builtin-memcpy',
      '-fno-builtin-memmove',
      '-fno-builtin-memcmp',
      '-fno-builtin-strlen',
      '-fno-builtin-strnlen',
      '-fno-builtin-strcpy',
      '-fno-builtin-strncpy',
      '-fno-builtin-strcmp',
      '-fno-builtin-strncmp',
      '-fno-builtin-strcasecmp',
      '-fno-builtin-strncasecmp',
      '-fno-builtin-vsprintf',
      '-fno-builtin-vsnprintf',
      '-fno-builtin-sprintf',
      '-fno-builtin-snprintf'
  ]

ob_objects = [lib_ob_server, lib_ob_rootserver, lib_ob_sql_simd, lib_ob_storage_simd, lib_ob_diagnose_lua, lib_ob_logservice, lib_ob_pl]
srcinc = include_directories('.')
# endfunction()
subdir('share')
subdir('sql')
subdir('pl')
subdir('rootserver')
subdir('logservice')
subdir('storage')
subdir('diagnose')
if 'BUILD_CDC_ONLY'
  subdir('observer')
  subdir('libtable')
else
  subdir('observer')
  subdir('libtable')
endif
subdir('plugin')

target_link_libraries(ob_share_static
  PUBLIC ob_share ob_share_extra)

libpangoft2_dep = declare_dependency(
    link_with: [lib_ob_share, lib_ob_share_extra],
    include_directories: './share')

