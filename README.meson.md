# oceanbase-meson

采用 meson 构建 oceanbase 数据库

尝试解决如下问题：

- 使 OB 成为一个不锁死架构和操作系统的通用项目
- 使用本地系统提供的库，做为构建依赖库
- 动态链接？

## 编译

默认使用 gcc 编译:

```sh
meson setup _build
meson compile -C _build/
```

使用 clang 编译:

```sh
CC=clang CXX=clang++ meson setup _build
meson compile -C _build/
```

配置编译静态库或动态库：

```sh
cd _build
meson configure -Ddefault_library=static #编译静态库
meson configure -Ddefault_library=shared #编译动态库
meson configure -Ddefault_library=both   #同时编译动态库和静态库
```
