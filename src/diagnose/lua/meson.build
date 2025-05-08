lua_sources = {
  'ob_lua_api.cpp',
  'ob_lua_handler.cpp'
}

lib_ob_diagnose_lua = library(
  'ob_diagnose_lua',
  sources : lua_sources,
  dependencies : [
    lua_dep,
    ],
  link_with : [lib_ob_base],
  if get_option('OB_BUILD_CCLS')
    compile_args: ['-DCCLS_LASY_OFF=1'],
  endif
)
