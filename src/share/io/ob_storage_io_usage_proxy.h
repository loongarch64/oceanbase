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

#ifndef OCEANBASE_LIB_OB_STORAGE_IO_USAGE_PROXY_H
#define OCEANBASE_LIB_OB_STORAGE_IO_USAGE_PROXY_H

#include "lib/ob_define.h"
namespace oceanbase
{
namespace common
{
class ObMySQLTransaction;
class ObString;
class ObMySQLProxy;
}
namespace share
{
class ObStorageIOUsageProxy
{
public:
    ObStorageIOUsageProxy() {}
    ~ObStorageIOUsageProxy() {}
    int update_storage_io_usage(common::ObMySQLTransaction &trans,
                                const uint64_t tenant_id,
                                const int64_t storage_id,
                                const int64_t dest_id,
                                const ObString &storage_mod,
                                const ObString &type,
                                const int64_t total);
private:
    DISALLOW_COPY_AND_ASSIGN(ObStorageIOUsageProxy);
};

}
}
#endif