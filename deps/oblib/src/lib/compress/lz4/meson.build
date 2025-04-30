lz4_171_sources = [
  'lz4_src/lz4_171.h',
  'lz4_src/lz4_171.c',
  'lz4_wrapper.h',
  'lz4_171_wrapper.cpp',
]

lib_lz4_171 = library(
  'lz4_171',
  sources : lz4_171_sources,
  include_directories : oblib_base_without_pass_dirs,
  cpp_args : [
    '-DLZ4LIB_VISIBILITY=',
    ],
  prelink: true,
  install: true,
  gnu_symbol_visibility: 'hidden',
)

objcopy_fake_lib_lz4_171 = ''
if filter_obj
  lib_target = lib_lz4_171
  if get_option('default_library') == 'both'
    lib_target = lib_lz4_171.get_static_lib()
  endif
  objcopy_fake_lib_lz4_171 = custom_target('liblz4_171',
    input: lib_target,
    output: 'liblz4_171.is-stripped',
    capture: true,
    command: [
      objcopy,
      '--localize-hidden',
      '@INPUT@',
    ],
    build_by_default: true,
  )
endif

lz4_191_sources = [
  'lz4_src/lz4_191.h',
  'lz4_src/lz4_191.c',
  'lz4_src/lz4hc_191.h',
  'lz4_src/lz4hc_191.c',
  'lz4_wrapper.h',
  'lz4_191_wrapper.cpp',
]

lib_lz4_191 = library(
  'lz4_191',
  sources : lz4_191_sources,
  include_directories : oblib_base_without_pass_dirs,
  cpp_args : [
    '-DLZ4LIB_VISIBILITY=',
    ],
  prelink: true,
  install: true,
  gnu_symbol_visibility: 'hidden',
)

objcopy_fake_lib_lz4_191 = ''
if filter_obj
  lib_target = lib_lz4_191
  if get_option('default_library') == 'both'
    lib_target = lib_lz4_191.get_static_lib()
  endif
  objcopy_fake_lib_lz4_191 = custom_target('liblz4_191',
    input: lib_target,
    output: 'liblz4_191.is-stripped',
    capture: true,
    command: [
      objcopy,
      '--localize-hidden',
      '@INPUT@',
    ],
    build_by_default: true,
  )
endif

lz4_sources = [
  'ob_lz4_compressor.cpp',
  'ob_lz4_compressor.h',
  'ob_lz4_compressor_191.cpp',
  'ob_lz4_stream_compressor.h',
  'ob_lz4_stream_compressor.cpp',
  'ob_lz4_stream_compressor_191.cpp',
  ]

lib_lz4 = library(
  'lz4',
  sources : lz4_sources,
  include_directories : oblib_base_without_pass_dirs,
  cpp_args : ['-fvisibility=hidden', '-DLZ4LIB_VISIBILITY=',
    '-Wundef',
    '-Werror=vla',
    '-Wno-maybe-uninitialized',
    '-Wno-missing-field-initializers',
    '-Wno-unused-parameter',
    '-Wunused-parameter',
    '-Wmaybe-uninitialized',
    '-Wno-error=deprecated-declarations'
    ],
  link_with: [
    lib_lz4_171,
    lib_lz4_191,
    ],
  name_prefix: '',
  prelink: true,
  install: true,
  gnu_symbol_visibility: 'hidden',
)

objcopy_fake_lib_lz4 = ''
if filter_obj
  lib_target = lib_lz4
  if get_option('default_library') == 'both'
    lib_target = lib_lz4.get_static_lib()
  endif
  objcopy_fake_lib_lz4 = custom_target('liblz4',
    input: lib_target,
    output: 'liblz4.is-stripped',
    capture: true,
    command: [
      objcopy,
      '--localize-hidden',
      '@INPUT@',
      ],
    build_by_default: true,
    )
endif
