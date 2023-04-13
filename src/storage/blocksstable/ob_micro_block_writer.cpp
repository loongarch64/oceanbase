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

#include "ob_micro_block_writer.h"
#include "lib/checksum/ob_crc64.h"
#include "storage/ob_i_store.h"

namespace oceanbase
{
using namespace common;
using namespace storage;
namespace blocksstable
{
ObMicroBlockWriter::ObMicroBlockWriter()
  :micro_block_size_limit_(0),
   column_count_(0),
   rowkey_column_count_(0),
   data_buffer_("MicrBlocWriter"),
   index_buffer_("MicrBlocWriter"),
   col_desc_array_(nullptr),
   need_calc_column_chksum_(false),
   is_inited_(false)
{
}

ObMicroBlockWriter::~ObMicroBlockWriter()
{
}

int ObMicroBlockWriter::init(
    const int64_t micro_block_size_limit,
    const int64_t rowkey_column_count,
    const int64_t column_count/* = 0*/,
    const common::ObIArray<share::schema::ObColDesc> *col_desc_array /* nullptr */,
    const bool need_calc_column_chksum/* = false*/)
{
  int ret = OB_SUCCESS;
  reset();
  if (OB_FAIL(check_input_param(micro_block_size_limit, column_count, rowkey_column_count))) {
    STORAGE_LOG(WARN, "micro block writer fail to check input param.", K(ret),
        K(micro_block_size_limit), K(column_count), K(rowkey_column_count));
  } else {
    micro_block_size_limit_ = micro_block_size_limit;
    rowkey_column_count_ = rowkey_column_count;
    column_count_ = column_count;
    need_calc_column_chksum_ = need_calc_column_chksum;
    need_check_lob_ = false;
    if (OB_NOT_NULL(col_desc_array_ = col_desc_array)) {
      for (int64_t i = 0; OB_SUCC(ret) && !need_check_lob_ && i < col_desc_array_->count(); i++) {
        need_check_lob_ = col_desc_array_->at(i).col_type_.is_lob_storage();
      }
    }
    is_inited_ = true;
  }
  return ret;
}

int ObMicroBlockWriter::inner_init()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!is_inited_)) {
    ret = OB_NOT_INIT;
    STORAGE_LOG(WARN, "not init", K(ret));
  } else if (data_buffer_.is_dirty()) {
    // has been inner_inited, do nothing
  } else if (OB_FAIL(data_buffer_.ensure_space(DEFAULT_DATA_BUFFER_SIZE))) {
    STORAGE_LOG(WARN, "data buffer fail to ensure space.", K(ret));
  } else if (OB_FAIL(index_buffer_.ensure_space(DEFAULT_INDEX_BUFFER_SIZE))) {
    STORAGE_LOG(WARN, "index buffer fail to ensure space.", K(ret));
  } else if (OB_FAIL(reserve_header(column_count_, rowkey_column_count_, need_calc_column_chksum_))) {
    STORAGE_LOG(WARN, "micro block writer fail to reserve header.",
        K(ret), K_(column_count));
  } else if (OB_FAIL(index_buffer_.write(static_cast<int32_t>(0)))) {
    STORAGE_LOG(WARN, "index buffer fail to write first offset.", K(ret));
  } else if (OB_UNLIKELY(data_buffer_.length() != get_data_base_offset()
        || index_buffer_.length() != get_index_base_offset())) {
    ret = OB_ERR_UNEXPECTED;
    STORAGE_LOG(WARN, "check length failed.", K(ret));
  }
  return ret;
}

int ObMicroBlockWriter::try_to_append_row(const int64_t &row_length)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(get_future_block_size(row_length) > block_size_upper_bound_)) {
    ret = OB_BUF_NOT_ENOUGH;
  }
  return ret;
}

int ObMicroBlockWriter::process_out_row_columns(const ObDatumRow &row)
{
  int ret = OB_SUCCESS;

  if (!need_check_lob_) {
  } else if (OB_UNLIKELY(nullptr == col_desc_array_ || row.get_column_count() != col_desc_array_->count())) {
    ret = OB_ERR_UNEXPECTED;
    STORAGE_LOG(WARN ,"unexpected column count not match", K(ret), K(need_check_lob_), K(row), KPC(col_desc_array_));
  } else if (!has_lob_out_row_) {
    for (int64_t i = 0; !has_lob_out_row_ && OB_SUCC(ret) && i < row.get_column_count(); ++i) {
      ObStorageDatum &datum = row.storage_datums_[i];
      if (col_desc_array_->at(i).col_type_.is_lob_storage()) {
        if (datum.is_nop() || datum.is_null()) {
        } else if (datum.len_ < sizeof(ObLobCommon)) {
          ret = OB_ERR_UNEXPECTED;
          STORAGE_LOG(WARN, "Unexpected lob datum len", K(ret), K(i), K(col_desc_array_->at(i).col_type_), K(datum));
        } else {
          const ObLobCommon &lob_common = datum.get_lob_data();
          has_lob_out_row_ = !lob_common.in_row_;
          STORAGE_LOG(DEBUG, "chaser debug lob out row", K(has_lob_out_row_), K(lob_common), K(datum));
        }
      }
    }
  }
  // uncomment this after varchar overflow supported
  //} else if (need_check_string_out) {
  //  if (!has_string_out_row_ && row.storage_datums_[i].is_outrow()) {
  //    has_string_out_row_ = true;
  //   }
  //}
  return ret;
}

int ObMicroBlockWriter::append_row(const ObDatumRow &row)
{
  int ret = OB_SUCCESS;
  int64_t pos = 0;
  if(!is_inited_) {
    ret = OB_NOT_INIT;
    STORAGE_LOG(WARN, "should init writer before append row", K(ret));
  } else if (!row.is_valid()) {
    ret = OB_INVALID_ARGUMENT;
    STORAGE_LOG(WARN, "row was invalid", K(row), K(ret));
  } else if (OB_FAIL(inner_init())) {
    STORAGE_LOG(WARN, "failed to inner init", K(ret));
  } else if (OB_FAIL(process_out_row_columns(row))) {
    STORAGE_LOG(WARN, "Failed to process out row columns", K(ret), K(row));
  } else {
    if (OB_UNLIKELY(row.get_column_count() != header_->column_count_)) {
      ret = OB_INVALID_ARGUMENT;
      STORAGE_LOG(WARN, "append row column count is not consistent with init column count",
          K(header_->column_count_), K(row.get_column_count()), K(ret));
    } else if (OB_FAIL(row_writer_.write(rowkey_column_count_, row, data_buffer_.current(),
            data_buffer_.remain(), pos))) {
      if (OB_BUF_NOT_ENOUGH != ret) {
        STORAGE_LOG(WARN, "row writer fail to write row.", K(ret), K(rowkey_column_count_), K(row),
            K(OB_P(data_buffer_.current())), K(OB_P(data_buffer_.remain())), K(pos));
      }
    } else if (is_exceed_limit(pos)) {
      STORAGE_LOG(DEBUG, "micro block exceed limit", K(pos),
          K(header_->row_count_), K(get_block_size()), K(micro_block_size_limit_));
      ret = OB_BUF_NOT_ENOUGH;
    } else if (OB_FAIL(try_to_append_row(pos))) {
      if (OB_UNLIKELY(OB_BUF_NOT_ENOUGH != ret)) {
        STORAGE_LOG(DEBUG, "fail to try append row", K(ret));
      }
    } else if (OB_FAIL(finish_row(pos))) {
      STORAGE_LOG(WARN, "micro block writer fail to finish row.", K(ret), K(pos));
    } else if (header_->has_column_checksum_
        && OB_FAIL(cal_column_checksum(row, header_->column_checksums_))) {
      STORAGE_LOG(WARN, "fail to cal column chksum", K(ret), K(row), KPC_(header));
    } else {
      cal_row_stat(row);
      if (need_cal_row_checksum()) {
        micro_block_checksum_ = cal_row_checksum(row, micro_block_checksum_);
      }
    }
  }
  return ret;
}

int ObMicroBlockWriter::build_block(char *&buf, int64_t &size)
{
  int ret = OB_SUCCESS;
  if(!is_inited_){
    ret = OB_NOT_INIT;
    STORAGE_LOG(WARN, "should init writer before append row", K(ret));
  } else if (OB_UNLIKELY(!data_buffer_.is_dirty())) {
    ret = OB_ERR_UNEXPECTED;
    STORAGE_LOG(WARN, "unexpected empty block", K(ret));
  } else {
    if (last_rows_count_ == header_->row_count_) {
      header_->single_version_rows_ = 1;
      STORAGE_LOG(DEBUG, "all rows are single version", K(last_rows_count_));
    }
    header_->row_index_offset_ = static_cast<int32_t>(data_buffer_.length());
    header_->contain_uncommitted_rows_ = contain_uncommitted_row_;
    header_->max_merged_trans_version_ = max_merged_trans_version_;
    header_->has_string_out_row_ = has_string_out_row_;
    header_->all_lob_in_row_ = !has_lob_out_row_;
    header_->is_last_row_last_flag_ = is_last_row_last_flag_;
    if (data_buffer_.remain() < get_index_size()) {
      ret = OB_SIZE_OVERFLOW;
      STORAGE_LOG(WARN, "row data buffer is overflow.",
          K(data_buffer_.remain()), K(get_index_size()), K(ret));
    } else if (OB_FAIL(data_buffer_.write(
            index_buffer_.data(), get_index_size()))) {
      STORAGE_LOG(WARN, "data buffer fail to write index.",
          K(ret), K(OB_P(index_buffer_.data())), K(get_index_size()));
    } else {
      buf = data_buffer_.data();
      size = data_buffer_.length();
    }
  }
  return ret;
}

int ObMicroBlockWriter::append_hash_index(ObMicroBlockHashIndexBuilder& hash_index_builder)
{
  int ret = OB_SUCCESS;
  header_->contains_hash_index_ = 0;
  if (hash_index_builder.is_valid()) {
    if (is_contain_uncommitted_row()) {
      ret = OB_NOT_SUPPORTED;
    } else if (OB_FAIL(hash_index_builder.build_block(data_buffer_))) {
      if (ret != OB_NOT_SUPPORTED) {
        STORAGE_LOG(WARN, "data buffer fail to write hash index.", K(ret));
      }
    } else {
      header_->contains_hash_index_ = 1;
      header_->hash_index_offset_from_end_ = hash_index_builder.estimate_size();
    }
  }
  return ret;
}

bool ObMicroBlockWriter::has_enough_space_for_hash_index(const int64_t hash_index_size) const {
  const int64_t total_size = get_data_size() + get_index_size() + hash_index_size;
  return total_size <= micro_block_size_limit_ && total_size <= block_size_upper_bound_;
}

void ObMicroBlockWriter::reset()
{
  ObIMicroBlockWriter::reuse();
  micro_block_size_limit_ = 0;
  column_count_ = 0;
  rowkey_column_count_ = 0;
  need_calc_column_chksum_ = false;
  row_writer_.reset();
  header_ = nullptr;
  data_buffer_.reset();
  index_buffer_.reset();
  col_desc_array_ = nullptr;
  is_inited_ = false;
}

void ObMicroBlockWriter::reuse()
{
  ObIMicroBlockWriter::reuse();
  row_writer_.reset();
  data_buffer_.reuse();
  index_buffer_.reuse();
  header_ = nullptr;
}

int ObMicroBlockWriter::check_input_param(
    const int64_t micro_block_size_limit,
    const int64_t column_count,
    const int64_t rowkey_column_count)
{
  int ret = OB_SUCCESS;
  if (micro_block_size_limit <= 0) {
    ret = OB_INVALID_ARGUMENT;
    STORAGE_LOG(WARN, "invalid micro block writer input argument.", K(micro_block_size_limit), K(ret));
  } else if (rowkey_column_count <= 0 ||
      (column_count <= 0 || column_count < rowkey_column_count)) {
    ret = OB_INVALID_ARGUMENT;
    STORAGE_LOG(WARN, "invalid micro block writer input argument.", K(ret), K(column_count),
                    K(rowkey_column_count));
  }
  return ret;
}

int ObMicroBlockWriter::finish_row(const int64_t length)
{
  int ret = OB_SUCCESS;
  if (!is_inited_) {
    ret = OB_NOT_INIT;
    STORAGE_LOG(WARN, "should init writer before finish row", K(ret));
  } else if (length <= 0) {
    ret = OB_INVALID_ARGUMENT;
    STORAGE_LOG(WARN, "length was invalid", K(length), K(ret));
  } else if (OB_FAIL(data_buffer_.advance(length))) {
    STORAGE_LOG(WARN, "data buffer fail to advance.", K(ret));
  } else {
    int32_t row_offset = static_cast<int32_t>(data_buffer_.length() - header_->header_size_);
    if (OB_FAIL(index_buffer_.write(row_offset))) {
      STORAGE_LOG(WARN, "index buffer fail to write row offset.", K(row_offset), K(ret));
    } else {
      ++ header_->row_count_;
    }
  }
  return ret;
}

int ObMicroBlockWriter::reserve_header(
    const int64_t column_count,
    const int64_t rowkey_column_count,
    const bool need_calc_column_chksum)
{
  int ret = OB_SUCCESS;

  if (column_count < 0) { // column_count of sparse row is 0
    ret = OB_INVALID_ARGUMENT;
    STORAGE_LOG(WARN, "column_count was invalid", K(column_count), K(ret));
  } else {
    const int32_t header_size = ObMicroBlockHeader::get_serialize_size(column_count, need_calc_column_chksum);
    header_ = reinterpret_cast<ObMicroBlockHeader*>(data_buffer_.data());

    if (OB_FAIL(data_buffer_.advance(header_size))) {
      STORAGE_LOG(WARN, "data buffer fail to advance header size.", K(ret), K(header_size));
    } else {
      MEMSET(header_, 0, header_size);
      header_->magic_ = MICRO_BLOCK_HEADER_MAGIC;
      header_->version_ = MICRO_BLOCK_HEADER_VERSION;
      header_->header_size_ = header_size;
      header_->column_count_ = static_cast<int32_t>(column_count);
      header_->rowkey_column_count_ = static_cast<int32_t>(rowkey_column_count);
      header_->row_store_type_ = FLAT_ROW_STORE;
      header_->has_column_checksum_ = need_calc_column_chksum;
      if (header_->has_column_checksum_) {
        header_->column_checksums_ = reinterpret_cast<int64_t *>(
            data_buffer_.data() + ObMicroBlockHeader::COLUMN_CHECKSUM_PTR_OFFSET);
      } else {
        header_->column_checksums_ = nullptr;
      }
    }
  }

  return ret;
}

bool ObMicroBlockWriter::is_exceed_limit(const int64_t row_length)
{
  return header_->row_count_ > 0 && get_future_block_size(row_length) > micro_block_size_limit_;
}

}//end namespace blocksstable
}//end namespace oceanbase

