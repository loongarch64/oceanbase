subdir('..')
obtable_dep = declare_dependency(link_with: lib_obtable_base_objects, include_directories: include_directories('../src'))

obtable_examples = [
  ['table_example', 'ob_table_example.cpp'],
  ['table_example_s', 'ob_table_example.cpp'],
  ['kvtable_example', 'ob_kvtable_example.cpp'],
  ['pstore_example', 'ob_pstore_example.cpp'],
  ['kvtable_benchmark', 'ob_kvtable_benchmark.cpp'],
]

foreach obtable_example: obtable_examples
  executable(
    obtable_example[0],
    obtable_example[1],
    include_directories: [
      include_directories('../src')
    ],
    link_args: [-Wl,--whole-archive -Wl,--no-whole-archive -Wl,--allow-multiple-definition],
    dependencies: obtable_dep,
    install: false,
  )
endforeach
