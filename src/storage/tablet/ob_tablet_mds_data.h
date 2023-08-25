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

#ifndef OCEANBASE_STORAGE_OB_TABLET_MDS_DATA
#define OCEANBASE_STORAGE_OB_TABLET_MDS_DATA

#include <stdint.h>
#include "lib/container/ob_se_array.h"
#include "lib/string/ob_string.h"
#include "share/ob_tablet_autoincrement_param.h"
#include "storage/compaction/ob_compaction_util.h"
#include "storage/multi_data_source/adapter_define/mds_dump_node.h"
#include "storage/tablet/ob_tablet_complex_addr.h"
#include "storage/tablet/ob_tablet_member_wrapper.h"
#include "storage/tablet/ob_tablet_full_medium_info.h"

namespace oceanbase
{
namespace common
{
class ObIAllocator;
class ObArenaAllocator;
}
namespace compaction
{
class ObMediumCompactionInfoKey;
class ObMediumCompactionInfo;
class ObMediumCompactionInfoList;
class ObExtraMediumInfo;
}

namespace storage
{
class ObTabletFullMemoryMdsData;

class ObTabletMdsDumpStruct
{
  OB_UNIS_VERSION(1);
public:
  ObTabletMdsDumpStruct();
  ~ObTabletMdsDumpStruct();
  ObTabletMdsDumpStruct(const ObTabletMdsDumpStruct &) = delete;
  ObTabletMdsDumpStruct &operator=(const ObTabletMdsDumpStruct &) = delete;
public:
  int init(common::ObIAllocator &allocator);
  void reset();
  int assign(const ObTabletMdsDumpStruct &other, common::ObIAllocator &allocator);

  TO_STRING_KV(K_(uncommitted_kv), K_(committed_kv));
public:
  ObTabletComplexAddr<mds::MdsDumpKV> uncommitted_kv_;
  ObTabletComplexAddr<mds::MdsDumpKV> committed_kv_;
};

class ObTabletMdsData
{
  friend class ObTablet;
public:
  ObTabletMdsData();
  ~ObTabletMdsData();
  ObTabletMdsData(const ObTabletMdsData&) = delete;
  ObTabletMdsData &operator=(const ObTabletMdsData&) = delete;
public:
  void reset();
  int init(common::ObIAllocator &allocator);
  int init(
      common::ObIAllocator &allocator,
      const ObTabletFullMemoryMdsData &full_memory_mds_data);
  int init(
      common::ObIAllocator &allocator,
      const ObTabletMdsData &mds_table_data,
      const ObTabletMdsData &base_data,
      const int64_t finish_medium_scn);
  int init(
      common::ObIAllocator &allocator,
      const ObTabletMdsData &other,
      const int64_t finish_medium_scn,
      const ObMergeType merge_type = ObMergeType::MERGE_TYPE_MAX);
  int init(
      common::ObIAllocator &allocator,
      const ObTabletMdsData &other,
      const ObTabletFullMediumInfo &full_memory_medium_info_list,
      const int64_t finish_medium_scn);
  int init_with_update_medium_info(
      common::ObIAllocator &allocator,
      const ObTabletMdsData &other);
  int init_empty_shell(const ObTabletCreateDeleteMdsUserData &tablet_status);
  int set_tablet_status(
      ObArenaAllocator *allocator,
      const ObTabletStatus::Status &tablet_status,
      const ObTabletMdsUserDataType &data_type);
  bool is_valid() const;
  void set_mem_addr();
public:
  int serialize(char *buf, const int64_t buf_len, int64_t &pos) const;
  int deserialize(const char *buf, const int64_t data_len, int64_t &pos);
  int64_t get_serialize_size() const;

  TO_STRING_KV(K_(is_inited),
               K_(tablet_status),
               K_(aux_tablet_info),
               K_(extra_medium_info),
               K_(medium_info_list),
               K_(auto_inc_seq),
               K_(tablet_status_cache),
               K_(aux_tablet_info_cache));
public:
  static int load_mds_dump_kv(
      common::ObIAllocator &allocator,
      const ObTabletComplexAddr<mds::MdsDumpKV> &complex_addr,
      const mds::MdsDumpKV *&kv);
  static int load_medium_info_list(
      common::ObIAllocator &allocator,
      const ObTabletComplexAddr<ObTabletDumpedMediumInfo> &complex_addr,
      const ObTabletDumpedMediumInfo *&medium_info_list);
  static void free_mds_dump_kv(
      common::ObIAllocator &allocator,
      const mds::MdsDumpKV *kv);
  static void free_medium_info_list(
      common::ObIAllocator &allocator,
      const ObTabletDumpedMediumInfo *medium_info_list);
  static int fetch_auto_inc_seq(
      const ObTabletComplexAddr<share::ObTabletAutoincSeq> &auto_inc_seq_addr,
      ObTabletMemberWrapper<share::ObTabletAutoincSeq> &wrapper);
  static int build_tablet_status(
      common::ObArenaAllocator &allocator,
      const ObTabletTxMultiSourceDataUnit &tx_data,
      const share::SCN &create_commit_scn,
      ObTabletMdsData &mds_data);
  static int build_aux_tablet_info(
      common::ObArenaAllocator &allocator,
      const ObTabletTxMultiSourceDataUnit &tx_data,
      const ObTabletBindingInfo &ddl_data,
      const share::SCN &clog_checkpoint_scn,
      ObTabletMdsData &mds_data);
  static int build_auto_inc_seq(
      common::ObArenaAllocator &allocator,
      const share::ObTabletAutoincSeq &auto_inc_seq,
      ObTabletMdsData &mds_data);
  static int build_mds_data(
    common::ObArenaAllocator &allocator,
    const share::ObTabletAutoincSeq &auto_inc_seq,
    const ObTabletTxMultiSourceDataUnit &tx_data,
    const share::SCN &create_commit_scn,
    const ObTabletBindingInfo &ddl_data,
    const share::SCN &clog_checkpoint_scn,
    const compaction::ObMediumCompactionInfoList &info_list,
    ObTabletMdsData &mds_data);
private:
  int alloc_and_new(common::ObIAllocator &allocator);
  int do_init(
      common::ObIAllocator &allocator,
      const mds::MdsDumpKV *tablet_status_uncommitted_kv,
      const mds::MdsDumpKV *tablet_status_committed_kv,
      const mds::MdsDumpKV *aux_tablet_info_uncommitted_kv,
      const mds::MdsDumpKV *aux_tablet_info_committed_kv,
      const share::ObTabletAutoincSeq *auto_inc_seq);
  int init_medium_info_list(
      common::ObIAllocator &allocator,
      const ObTabletDumpedMediumInfo *old_medium_info_list,
      const compaction::ObExtraMediumInfo &old_extra_medium_info,
      const int64_t finish_medium_scn = 0,
      const ObMergeType merge_type = ObMergeType::MERGE_TYPE_MAX);
  int init_medium_info_list(
      common::ObIAllocator &allocator,
      const ObTabletDumpedMediumInfo *old_medium_info_list,
      const ObTabletFullMediumInfo &full_memory_medium_info_list,
      const compaction::ObExtraMediumInfo &old_extra_medium_info,
      const int64_t finish_medium_scn);
  int init_with_update_medium_info(
      common::ObIAllocator &allocator,
      const ObTabletDumpedMediumInfo *old_medium_info_list,
      const compaction::ObExtraMediumInfo &old_extra_medium_info);
  static int fuse_mds_dump_node(
      common::ObIAllocator &allocator,
      const ObTabletComplexAddr<mds::MdsDumpKV> &mds_table_data,
      const ObTabletComplexAddr<mds::MdsDumpKV> &base_data,
      ObTabletComplexAddr<mds::MdsDumpKV> &fused_data);
  static int fuse_mds_dump_node(
      common::ObIAllocator &allocator,
      const ObTabletMdsDumpStruct &mds_table_data,
      const ObTabletMdsDumpStruct &base_data,
      ObTabletMdsDumpStruct &fused_data);
  static int fuse_mds_dump_node(
      common::ObIAllocator &allocator,
      const ObTabletComplexAddr<share::ObTabletAutoincSeq> &mds_table_data,
      const ObTabletComplexAddr<share::ObTabletAutoincSeq> &base_data,
      ObTabletComplexAddr<share::ObTabletAutoincSeq> &fused_data);
  static int fuse_mds_dump_node(
      common::ObIAllocator &allocator,
      const int64_t finish_medium_scn,
      const ObTabletComplexAddr<ObTabletDumpedMediumInfo> &mds_table_data,
      const ObTabletComplexAddr<ObTabletDumpedMediumInfo> &base_data,
      ObTabletComplexAddr<ObTabletDumpedMediumInfo> &fused_data);
  static int read_medium_info(
      common::ObIAllocator &allocator,
      const ObMetaDiskAddr &addr,
      common::ObSEArray<compaction::ObMediumCompactionInfo*, 1> &array);
  static int copy_medium_info_list(
      const int64_t finish_medium_scn,
      const ObTabletDumpedMediumInfo &input_medium_info_list,
      ObTabletDumpedMediumInfo &medium_info_list);
  static int copy_medium_info_list(
      const int64_t finish_medium_scn,
      const ObTabletDumpedMediumInfo &input_medium_info_list1,
      const ObTabletDumpedMediumInfo &input_medium_info_list2,
      ObTabletDumpedMediumInfo &medium_info_list);
  template <typename T>
  static int update_user_data_from_complex_addr(
      const ObTabletComplexAddr<mds::MdsDumpKV> &complex_addr,
      T &user_data);
public:
  bool is_inited_;
  ObTabletMdsDumpStruct tablet_status_;
  ObTabletMdsDumpStruct aux_tablet_info_;

  compaction::ObExtraMediumInfo extra_medium_info_;
  ObTabletComplexAddr<ObTabletDumpedMediumInfo> medium_info_list_;

  ObTabletComplexAddr<share::ObTabletAutoincSeq> auto_inc_seq_;
  ObTabletCreateDeleteMdsUserData tablet_status_cache_;
  ObTabletBindingMdsUserData aux_tablet_info_cache_;
};

template <typename T>
int ObTabletMdsData::update_user_data_from_complex_addr(
    const ObTabletComplexAddr<mds::MdsDumpKV> &complex_addr,
    T &user_data)
{
  int ret = common::OB_SUCCESS;

  if (OB_UNLIKELY(!complex_addr.is_memory_object())) {
    ret = common::OB_INVALID_ARGUMENT;
    STORAGE_LOG(WARN, "complex addr is not memory type", K(ret), K(complex_addr));
  } else {
    const mds::MdsDumpKV *kv = complex_addr.ptr_;
    const common::ObString &str = kv->v_.user_data_;
    if (str.empty()) {
      // do nothing
    } else {
      int64_t pos = 0;
      if (OB_FAIL(user_data.deserialize(str.ptr(), str.length(), pos))) {
        STORAGE_LOG(WARN, "failed to deserialize", K(ret));
      }
    }
  }

  return ret;
}
} // namespace storage
} // namespace oceanbase

#endif // OCEANBASE_STORAGE_OB_TABLET_MDS_DATA
