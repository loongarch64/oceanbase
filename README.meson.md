# oceanbase-meson

采用 meson 构建的 oceanbase 数据库

解决如下问题：

- 使用本机依赖库
- 动态链接？

## 编译

```sh
meson setup _build
meson compile -C _build/
```
