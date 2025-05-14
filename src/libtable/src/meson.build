
obtable_base_objects_sources = {
  'libobtable.cpp',
  'ob_hkv_table.cpp',
  'ob_tablet_location_proxy.cpp',
  'ob_pstore.cpp',
  'ob_table_impl.cpp',
  'ob_table_rpc_impl.cpp',
  'ob_table_service_client.cpp'
}

lib_obtable_base_objects = library(
  'obtable_base_objects',
  sources : obtable_base_objects_sources,
  link_with : [lib_oceanbase, hyperscan_lib],
  if get_option('OB_BUILD_CCLS')
    compile_args: ['-DCCLS_LASY_OFF=1'],
  endif
)

GIT_REVISION = run_command ('git', '-C', meson.source_root(), 'rev-parse', 'HEAD')
GIT_BRANCH = run_command ('git', '-C', meson.source_root(), 'rev-parse', '--abbrev-ref', 'HEAD')
file_data = configuration_data()
file_data.set('BUILD_FLAGS', get_option ('BUILD_TYPE'))
file_data.set('BUILD_NUMBER', '1')
file_data.set('GIT_REVISION', GIT_REVISION)
file_data.set('GIT_BRANCH', GIT_BRANCH)
file_data.set('BUILD_INFO', 'xxxxxxx')

file_in_file = configure_file(
  input : join_paths(sources_root, 'src', 'share', 'ob_version.cpp.in'),
  output : 'ob_version.cpp',
  configuration : file_data
)
lib_obtable_static = static_library(
  'obtable_static',
  sources: 'ob_version.cpp'
  link_with : [ob_objects, oblib_object_libraries],
  )
