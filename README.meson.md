# oceanbase-meson

采用 meson 构建 oceanbase 数据库

尝试解决如下问题：

- 使 OB 成为一个不锁死架构和操作系统的通用项目
- 使用本地系统提供的库，做为构建依赖库
- 动态链接？

## 编译

```sh
meson setup _build
meson compile -C _build/
```
