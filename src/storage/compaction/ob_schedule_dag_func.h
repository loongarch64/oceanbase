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

#ifndef OCEANBASE_STORAGE_COMPACTION_OB_SCHEDULE_DAG_FUNC_H_
#define OCEANBASE_STORAGE_COMPACTION_OB_SCHEDULE_DAG_FUNC_H_
#include "lib/container/ob_iarray.h"
#include "storage/compaction/ob_compaction_util.h"

namespace oceanbase
{
namespace share
{
class ObLSID;
}
namespace storage
{
namespace mds
{
class ObMdsTableMergeDagParam;
}
struct ObDDLTableMergeDagParam;
}

namespace compaction
{
struct ObTabletMergeDagParam;
struct ObCOMergeDagParam;
struct ObTabletSchedulePair;
struct ObBatchFreezeTabletsParam;
#ifdef OB_BUILD_SHARED_STORAGE
struct ObTabletsRefreshSSTableParam;
struct ObVerifyCkmParam;
struct ObUpdateSkipMajorParam;
#endif

class ObScheduleDagFunc
{
public:
  static int schedule_tablet_merge_dag(
      ObTabletMergeDagParam &param,
      const bool is_emergency = false);
  static int schedule_tx_table_merge_dag(
      ObTabletMergeDagParam &param,
      const bool is_emergency = false);
  static int schedule_ddl_table_merge_dag(
      storage::ObDDLTableMergeDagParam &param,
      const bool is_emergency = false);
  static int schedule_tablet_co_merge_dag_net(
      ObCOMergeDagParam &param);
  static int schedule_mds_table_merge_dag(
      storage::mds::ObMdsTableMergeDagParam &param,
      const bool is_emergency = false);
  static int schedule_batch_freeze_dag(
    const ObBatchFreezeTabletsParam &freeze_param);
#ifdef OB_BUILD_SHARED_STORAGE
  static int schedule_tablet_refresh_dag(
      ObTabletsRefreshSSTableParam &param,
      const bool is_emergency = false);
  static int schedule_verify_ckm_dag(ObVerifyCkmParam &param);
  static int schedule_update_skip_major_tablet_dag(const ObUpdateSkipMajorParam &param);
#endif
  static int get_exec_mode(
      const bool is_major_merge_type,
      const share::ObLSID &ls_id,
      ObExecMode &exec_mode);
};

}
} /* namespace oceanbase */

#endif /* OCEANBASE_STORAGE_OB_SCHEDULE_DAG_FUNC_H_ */
