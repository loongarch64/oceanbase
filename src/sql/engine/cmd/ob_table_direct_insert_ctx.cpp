/**
 * Copyright (c) 2023 OceanBase
 * OceanBase CE is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 */

#define USING_LOG_PREFIX SQL_ENG

#include "sql/engine/cmd/ob_table_direct_insert_ctx.h"
#include "observer/omt/ob_tenant.h"
#include "observer/table_load/ob_table_load_exec_ctx.h"
#include "observer/table_load/ob_table_load_instance.h"
#include "observer/table_load/ob_table_load_schema.h"
#include "observer/table_load/ob_table_load_service.h"
#include "observer/table_load/ob_table_load_struct.h"
#include "sql/engine/ob_exec_context.h"

namespace oceanbase
{
using namespace common;
using namespace observer;
using namespace storage;
using namespace share;

namespace sql
{
ObTableDirectInsertCtx::~ObTableDirectInsertCtx()
{
  destroy();
}

int ObTableDirectInsertCtx::init(
    ObExecContext *exec_ctx,
    const uint64_t table_id,
    const int64_t parallel,
    const bool is_incremental,
    const bool enable_inc_replace)
{
  int ret = OB_SUCCESS;
  const uint64_t tenant_id = MTL_ID();
  ObSQLSessionInfo *session_info = nullptr;
  ObSchemaGetterGuard *schema_guard = nullptr;
  if (IS_INIT) {
    ret = OB_INIT_TWICE;
    LOG_WARN("ObTableDirectInsertCtx init twice", KR(ret));
  } else if (OB_ISNULL(exec_ctx)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("exec_ctx cannot be null", KR(ret));
  } else if (OB_ISNULL(session_info = exec_ctx->get_my_session())) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected session info is null", KR(ret));
  } else if (OB_ISNULL(exec_ctx->get_sql_ctx())) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected sql ctx is null", KR(ret));
  } else if (OB_ISNULL(schema_guard = exec_ctx->get_sql_ctx()->schema_guard_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected schema guard is null", KR(ret));
  } else if (OB_UNLIKELY(session_info->get_ddl_info().is_mview_complete_refresh() && enable_inc_replace)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected mview complete refresh enable inc replace", KR(ret));
  } else {
    is_direct_ = true;
    if (OB_ISNULL(load_exec_ctx_ = OB_NEWx(ObTableLoadSqlExecCtx, &exec_ctx->get_allocator()))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("fail to new ObTableLoadSqlExecCtx", KR(ret));
    } else if (OB_ISNULL(table_load_instance_ =
                           OB_NEWx(ObTableLoadInstance, &exec_ctx->get_allocator()))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("fail to new ObTableLoadInstance", KR(ret));
    } else {
      load_exec_ctx_->exec_ctx_ = exec_ctx;
      ObArray<uint64_t> column_ids;
      omt::ObTenant *tenant = nullptr;
      ObDirectLoadMethod::Type method = (is_incremental ? ObDirectLoadMethod::INCREMENTAL : ObDirectLoadMethod::FULL);
      ObDirectLoadInsertMode::Type insert_mode = ObDirectLoadInsertMode::INVALID_INSERT_MODE;
      if (session_info->get_ddl_info().is_mview_complete_refresh()) {
        insert_mode = ObDirectLoadInsertMode::OVERWRITE;
      } else if (enable_inc_replace) {
        insert_mode = ObDirectLoadInsertMode::INC_REPLACE;
      } else {
        insert_mode = ObDirectLoadInsertMode::NORMAL;
      }
      if (OB_FAIL(GCTX.omt_->get_tenant(MTL_ID(), tenant))) {
        LOG_WARN("fail to get tenant handle", KR(ret), K(MTL_ID()));
      } else if (OB_FAIL(ObTableLoadService::check_support_direct_load(*schema_guard,
                                                                       table_id,
                                                                       method,
                                                                       insert_mode))) {
        LOG_WARN("fail to check support direct load", KR(ret));
      } else if (OB_FAIL(ObTableLoadSchema::get_column_ids(*schema_guard,
                                                           tenant_id,
                                                           table_id,
                                                           column_ids))) {
        LOG_WARN("failed to init store column idxs", KR(ret));
      } else {
        ObTableLoadParam param;
        param.tenant_id_ = MTL_ID();
        param.table_id_ = table_id;
        param.batch_size_ = 100;
        param.parallel_ = parallel;
        param.session_count_ = parallel;
        param.column_count_ = column_ids.count();
        param.px_mode_ = true;
        param.online_opt_stat_gather_ = true;
        param.need_sort_ = true;
        param.max_error_row_count_ = 0;
        param.dup_action_ = (enable_inc_replace ? sql::ObLoadDupActionType::LOAD_REPLACE
                                                : sql::ObLoadDupActionType::LOAD_STOP_ON_DUP);
        param.online_opt_stat_gather_ = is_online_gather_statistics_;
        param.method_ = method;

        param.insert_mode_ = insert_mode;
        if (OB_FAIL(table_load_instance_->init(param, column_ids, load_exec_ctx_))) {
          LOG_WARN("failed to init direct loader", KR(ret));
        } else {
          is_inited_ = true;
          LOG_DEBUG("succeeded to init direct loader", K(param));
        }
      }
    }
  }
  return ret;
}

// commit() should be called before finish()
int ObTableDirectInsertCtx::commit()
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
    LOG_WARN("ObTableDirectInsertCtx is not init", KR(ret));
  } else if (OB_FAIL(table_load_instance_->px_commit_data())) {
    LOG_WARN("failed to do px_commit_data", KR(ret));
  }
  return ret;
}

// finish() should be called after commit()
int ObTableDirectInsertCtx::finish()
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
    LOG_WARN("ObTableDirectInsertCtx is not init", KR(ret));
  } else if (OB_FAIL(table_load_instance_->px_commit_ddl())) {
    LOG_WARN("failed to do px_commit_ddl", KR(ret));
  } else {
    LOG_DEBUG("succeeded to finish direct loader");
  }
  return ret;
}

void ObTableDirectInsertCtx::destroy()
{
  if (OB_NOT_NULL(table_load_instance_)) {
    table_load_instance_->~ObTableLoadInstance();
    table_load_instance_ = nullptr;
  }
  if (OB_NOT_NULL(load_exec_ctx_)) {
    load_exec_ctx_->~ObTableLoadSqlExecCtx();
    load_exec_ctx_ = nullptr;
  }
  is_inited_ = false;
  is_direct_ = false;
  is_online_gather_statistics_ = false;
}

} // namespace sql
} // namespace oceanbase
