/**
 * Copyright (c) 2021 OceanBase
 * OceanBase CE is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 */

#ifndef __OB_COMMON_SQLCLIENT_MYSQL_PREPARED_PARAM__
#define __OB_COMMON_SQLCLIENT_MYSQL_PREPARED_PARAM__

#include <mysql/mysql.h>
#include "lib/string/ob_string.h"
#include "lib/mysqlclient/ob_mysql_connection.h"
#include "lib/mysqlclient/ob_mysql_result.h"

namespace oceanbase {
namespace common {
class ObIAllocator;
namespace sqlclient {
class ObMySQLPreparedStatement;
class ObMySQLPreparedParam {
public:
  explicit ObMySQLPreparedParam(ObMySQLPreparedStatement& stmt);
  ~ObMySQLPreparedParam();
  int init();
  int bind_param();
  void close();

  /*
   * bind data output buffer
   */
  int bind_param(
      const int64_t col_idx, enum_field_types buf_type, char* out_buf, const int64_t buf_len, unsigned long& res_len);

private:
  ObMySQLPreparedStatement& stmt_;
  common::ObIAllocator& alloc_;
  int64_t param_count_;
  MYSQL_BIND* bind_;
};
}  // namespace sqlclient
}  // namespace common
}  // namespace oceanbase
#endif
