subdir('io')
subdir('memory')
subdir('packet')
subdir('thread')
subdir('util')

libeasy = library(
  'easy',
  sources : [
    io_sources,
    memory_sources,
    packet_sources,
    thread_sources,
    util_sources
  ],
  include_directories : easy_include_dirs,
  dependencies : [ easy_deps ],
  c_args : easy_base_c_args,
  install: true,
)

#libeasy_dep = declare_dependency(
#  link_with : libeasy,
#  dependencies : libm,
#  include_directories : easy_include_dirs,
#)
