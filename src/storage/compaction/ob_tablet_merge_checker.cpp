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

#include "storage/compaction/ob_tablet_merge_checker.h"
#include "lib/oblog/ob_log.h"
#include "lib/ob_errno.h"
#include "storage/compaction/ob_compaction_util.h"
#include "storage/tablet/ob_tablet.h"
#include "storage/ls/ob_ls.h"
#include "rootserver/ob_tenant_info_loader.h"

#define USING_LOG_PREFIX STORAGE_COMPACTION

using namespace oceanbase::common;
using namespace oceanbase::storage;

namespace oceanbase
{
namespace compaction
{
int ObTabletMergeChecker::check_need_merge(const ObMergeType merge_type, const ObTablet &tablet)
{
  int ret = OB_SUCCESS;
  bool need_merge = true;

  if (OB_UNLIKELY(merge_type <= ObMergeType::INVALID_MERGE_TYPE
      || merge_type >= ObMergeType::MERGE_TYPE_MAX)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("merge type is invalid", K(ret), "merge_type", merge_type_to_str(merge_type));
  } else if (!is_minor_merge(merge_type)
      && !is_mini_merge(merge_type)
      && !is_major_or_meta_merge_type(merge_type)
      && !is_medium_merge(merge_type)) {
    need_merge = true;
  } else {
    const share::ObLSID &ls_id = tablet.get_tablet_meta().ls_id_;
    const common::ObTabletID &tablet_id = tablet.get_tablet_meta().tablet_id_;
    bool is_empty_shell = tablet.is_empty_shell();
    if (is_minor_merge(merge_type) || is_mini_merge(merge_type)) {
      need_merge = !is_empty_shell;
    } else if (is_major_or_meta_merge_type(merge_type)) {
      need_merge = tablet.is_data_complete();
    }

    if (OB_FAIL(ret)) {
    } else if (!need_merge) {
      ret = OB_NO_NEED_MERGE;
      LOG_INFO("tablet has no need to merge", K(ret), K(ls_id), K(tablet_id),
          "merge_type", merge_type_to_str(merge_type), K(is_empty_shell));
    }
  }

  return ret;
}

int ObTabletMergeChecker::check_could_merge_for_medium(
  const ObTablet &tablet,
  bool &could_schedule_merge)
{
  int ret = OB_SUCCESS;
  ObTabletCreateDeleteMdsUserData user_data;
  bool committed_flag = false;
  could_schedule_merge = true;
  if (OB_FAIL(tablet.ObITabletMdsInterface::get_latest_tablet_status(user_data, committed_flag))) {
    LOG_WARN("failed to get tablet status", K(ret), K(tablet), K(user_data));
  } else if (ObTabletStatus::TRANSFER_OUT == user_data.tablet_status_
    || ObTabletStatus::TRANSFER_OUT_DELETED == user_data.tablet_status_) {
    could_schedule_merge = false;
    if (REACH_TENANT_TIME_INTERVAL(PRINT_LOG_INVERVAL)) {
      LOG_INFO("tablet status is TRANSFER_OUT or TRANSFER_OUT_DELETED, merging is not allowed", K(user_data), K(tablet));
    }
  }
  return ret;
}

int ObTabletMergeChecker::check_ls_state(ObLS &ls, bool &need_merge)
{
  int ret = OB_SUCCESS;
  need_merge = false;
  if (ls.is_deleted()) {
    if (REACH_TENANT_TIME_INTERVAL(PRINT_LOG_INVERVAL)) {
      LOG_INFO("ls is deleted", K(ret), K(ls));
    }
  } else if (ls.is_offline()) {
    if (REACH_TENANT_TIME_INTERVAL(PRINT_LOG_INVERVAL)) {
      LOG_INFO("ls is offline", K(ret), K(ls));
    }
  } else {
    need_merge = true;
  }
  return ret;
}

int ObTabletMergeChecker::check_ls_state_in_major(ObLS &ls, bool &need_merge)
{
  int ret = OB_SUCCESS;
  need_merge = false;
  bool is_remote_tenant = false;
  ObLSRestoreStatus restore_status;
  if (OB_FAIL(check_mtl_tenant_is_remote(is_remote_tenant))) {
    LOG_WARN("fail to check tenant is remote", K(ret));
  } else if (is_remote_tenant) {
    LOG_INFO("tenant restore data mode is remote, should not loop tablet to schedule", K(ret), "tenant_id", MTL_ID());
  } else if (OB_FAIL(check_ls_state(ls, need_merge))) {
    LOG_WARN("failed to check ls state", KR(ret), "ls_id", ls.get_ls_id());
  } else if (!need_merge) {
    // do nothing
  } else if (OB_FAIL(ls.get_ls_meta().get_restore_status(restore_status))) {
    LOG_WARN("failed to get restore status", K(ret), K(ls));
  } else if (OB_UNLIKELY(!restore_status.is_none())) {
    if (REACH_TENANT_TIME_INTERVAL(PRINT_LOG_INVERVAL)) {
      LOG_INFO("ls is in restore status, should not loop tablet to schedule", K(ret), "ls_id", ls.get_ls_id());
    }
  } else {
    need_merge = true;
  }
  return ret;
}

bool ObTabletMergeChecker::check_weak_read_ts_ready(
    const int64_t &merge_version,
    ObLS &ls)
{
  bool is_ready_for_compaction = false;
  SCN weak_read_scn;

  if (FALSE_IT(weak_read_scn = ls.get_ls_wrs_handler()->get_ls_weak_read_ts())) {
  } else if (weak_read_scn.get_val_for_tx() < merge_version) {
    FLOG_INFO("current slave_read_ts is smaller than freeze_ts, try later",
              "ls_id", ls.get_ls_id(), K(merge_version), K(weak_read_scn));
  } else {
    is_ready_for_compaction = true;
  }
  return is_ready_for_compaction;
}

int ObTabletMergeChecker::check_mtl_tenant_is_remote(bool &is_remote)
{
  int ret = OB_SUCCESS;
  is_remote = false;
  ObRestoreDataMode restore_data_mode;
  rootserver::ObTenantInfoLoader *tenant_info_loader = MTL(rootserver::ObTenantInfoLoader*);
  if (OB_ISNULL(tenant_info_loader)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("tenant info loader is null", K(ret));
  } else if (OB_FAIL(tenant_info_loader->get_restore_data_mode(restore_data_mode))) {
    if (REACH_TIME_INTERVAL(5 * 1000 * 1000)) {
      LOG_WARN("get restore_data_mode failed", K(ret));
    }
  } else if (restore_data_mode.is_remote_mode()) {
    is_remote = true;
  }
  return ret;
}

} // namespace compaction
} // namespace oceanbase
