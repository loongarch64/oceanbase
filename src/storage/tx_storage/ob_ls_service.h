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

#ifndef OCEABASE_STORAGE_LS_SERVICE_
#define OCEABASE_STORAGE_LS_SERVICE_

#include "lib/guard/ob_shared_guard.h"
#include "lib/allocator/ob_concurrent_fifo_allocator.h"          // ObConcurrentFIFOAllocator
#include "storage/tx_storage/ob_ls_map.h"
#include "storage/ob_storage_rpc.h"
#include "storage/ls/ob_ls_meta_package.h"                      // ObLSMetaPackage

namespace oceanbase
{
namespace observer
{
class ObIMetaReport;
}
namespace share
{
class ObLSID;
}
namespace blocksstable
{
class ObBaseStorageLogger;
class ObSuperBlockMetaEntry;
}

namespace storage
{
class ObLS;
class ObLSIterator;
class ObLSHandle;
struct ObLSMeta;

// Maintain the tenant <-> log streams mapping relationship
// Support log stream meta persistent and checkpoint
class ObLSService
{
public:
  ObLSService();
  ~ObLSService();

  static int mtl_init(ObLSService* &ls_service);
  int init(const uint64_t tenant_id,
           observer::ObIMetaReport *reporter);
  int start();
  int stop();
  void destroy();
  bool safe_to_destroy();
  void inc_ls_safe_destroy_task_cnt();
  void dec_ls_safe_destroy_task_cnt();
public:
  // create a LS
  // @param [in] arg, all the parameters that is need to create a LS.
  int create_ls(const obrpc::ObCreateLSArg &arg);
  // delete a LS
  // @param [in] ls_id, which LS is to be removed.
  int remove_ls(const share::ObLSID &ls_id, const bool is_replay);
  // create a LS for HighAvaiable
  // @param [in] meta_package, all the parameters that is needed to create a LS for ha
  int create_ls_for_ha(const share::ObTaskId task_id, const ObMigrationOpArg &arg);

  // create a LS for replay or update LS's meta
  // @param [in] ls_meta, all the parameters that is needed to create a LS for replay
  int replay_create_ls(const ObLSMeta &ls_meta);
  // replay create ls commit slog.
  // @param [in] ls_id, the create process of which is committed.
  int replay_create_ls_commit(const share::ObLSID &ls_id);
  // create a LS for replay or update LS's meta
  // @param [in] ls_meta, all the parameters that is needed to create a LS for replay
  int replay_update_ls(const ObLSMeta &ls_meta);
  // update LS's meta for restore
  // @param [in] meta_package, all the parameters that is needed to for restore
  int restore_update_ls(const ObLSMetaPackage &meta_package);
  // set a LS to REMOVED state and gc it later.
  // for replay create ls abort or remove
  // @param [in] ls_id, which ls need to be set REMOVED.
  int replay_remove_ls(const share::ObLSID &ls_id);

  // @param [in] ls_id, which ls does we need, mod is the module to get ls
  // @param [out] handle, a guard of the specified logsream.
  int get_ls(const share::ObLSID &ls_id,
             ObLSHandle &handle,
             ObLSGetMod mod);
  // @param [in] func, iterate all ls diagnose info
  int iterate_diagnose(const ObFunction<int(const storage::ObLS &ls)> &func);

  // remove the ls that is creating and write abort slog.
  int gc_ls_after_replay_slog();
  // enable replay all ls clog
  int enable_replay();

  // check whether a ls exist or not.
  // @param [in] ls_id, the ls we will check.
  // @param [out] exist, true if the ls exist, else false.
  int check_ls_exist(const share::ObLSID &ls_id,
                     bool &exist);
  // check whether a ls waiting for destroy or not.
  // @param [in] ls_id, the ls we will check.
  // @param [out] waiting, true if the ls waiting for destroy, else false.
  int check_ls_waiting_safe_destroy(const share::ObLSID &ls_id,
                                    bool &waiting);
  // get a log stream iterator.
  // @param [out] guard, the iterator created.
  // use guard just like a pointer of ObLSIterator
  int get_ls_iter(common::ObSharedGuard<ObLSIterator> &guard, ObLSGetMod mod);

  // tablet operation
  int create_tablet(const obrpc::ObBatchCreateTabletArg &batch_arg,
                    obrpc::ObCreateTabletBatchRes &result);
  // remove tablets from a ls
  // @param [in] arg, all the remove parameters needed.
  // @param [out] result, the return code of the remove op.
  int remove_tablet(const obrpc::ObBatchRemoveTabletArg &batch_arg,
                    obrpc::ObRemoveTabletRes &result);

  // tablet operation in transactions
  // Create tablets for a ls
  // @param [in] tx_desc, trans descriptor
  // @param [in] arg, all the create parameters needed.
  // @param [out] result, the return code and trans result of the op.
  int create_tablets_in_trans(transaction::ObTxDesc &tx_desc,
                              const obrpc::ObBatchCreateTabletArg &batch_arg,
                              obrpc::ObCreateTabletBatchInTransRes &result);

  // remove tablets from a ls
  // @param [in] tx_desc, trans descriptor
  // @param [in] arg, all the remove parameters needed.
  // @param [out] result, the return code of the remove op.
  int remove_tablets_in_trans(transaction::ObTxDesc &tx_desc,
                              const obrpc::ObBatchRemoveTabletArg &batch_arg,
                              obrpc::ObRemoveTabletsInTransRes &result);

  obrpc::ObStorageRpcProxy *get_storage_rpc_proxy() { return &storage_svr_rpc_proxy_; }
  storage::ObStorageRpc *get_storage_rpc() { return &storage_rpc_; }
  ObLSMap *get_ls_map() { return &ls_map_; }

  TO_STRING_KV(K_(tenant_id), K_(is_inited));
private:
  enum class ObLSCreateState {
      CREATE_STATE_INIT = 0, // begin
      CREATE_STATE_INNER_CREATED = 1, // inner_create_ls_ succ
      CREATE_STATE_ADDED_TO_MAP = 2, // add_ls_to_map_ succ
      CREATE_STATE_WRITE_PREPARE_SLOG = 3, // write_prepare_create_ls_slog_ succ
      CREATE_STATE_PALF_ENABLED = 4, // enable_palf succ
      CREATE_STATE_FINISH
  };
  int inner_create_ls_(const share::ObLSID &lsid,
                       const ObReplicaType replica_type,
                       const ObMigrationStatus &migration_status,
                       const share::ObLSRestoreStatus &restore_status,
                       const share::SCN &create_scn,
                       ObLS *&ls);
  int inner_del_ls_(ObLS *&ls);
  int add_ls_to_map_(ObLS *ls);
  int write_prepare_create_ls_slog_(const ObLSMeta &ls_meta) const;
  int write_commit_create_ls_slog_(const share::ObLSID &ls_id) const;
  int write_abort_create_ls_slog_(const share::ObLSID &ls_id) const;
  int write_remove_ls_slog_(const share::ObLSID &ls_id) const;
  int remove_ls_from_map_(const share::ObLSID &ls_id);
  void remove_ls_(ObLS *ls, const bool remove_from_disk = true);
  int replay_update_ls_(const ObLSMeta &ls_meta);
  int restore_update_ls_(const ObLSMetaPackage &meta_package);
  int replay_remove_ls_(const share::ObLSID &ls_id);
  int replay_create_ls_(const ObLSMeta &ls_meta);
  void del_ls_after_create_ls_failed_(ObLSCreateState& ls_create_state, ObLS *ls);

  int alloc_ls_(ObLS *&ls);
  bool is_ls_to_restore_(const obrpc::ObCreateLSArg &arg) const;
  bool need_create_inner_tablets_(const obrpc::ObCreateLSArg &arg) const;
  int get_restore_status_(
      share::ObLSRestoreStatus &restore_status);

private:
  bool is_inited_;
  bool is_running_;
  uint64_t tenant_id_;
  // a map from ls id to ls
  ObLSMap ls_map_;

  common::ObConcurrentFIFOAllocator ls_allocator_;
  common::ObConcurrentFIFOAllocator iter_allocator_;
  // protect the create and remove process
  lib::ObMutex change_lock_;
  observer::ObIMetaReport *rs_reporter_;

  //TOD(muwei.ym) src rpc framework should be tenant level
  obrpc::ObStorageRpcProxy storage_svr_rpc_proxy_;
  storage::ObStorageRpc storage_rpc_;

  // for safe destroy
  // store the ls is removing
  int64_t safe_ls_destroy_task_cnt_;
  DISALLOW_COPY_AND_ASSIGN(ObLSService);
};


}
}
#endif
