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

#ifndef OCEANBASE_LIB_OB_NAME_DEF_H_
#define OCEANBASE_LIB_OB_NAME_DEF_H_

// define common names to be used in logs (e.g. to_string, databuff_xxx and ObPhyOperator::to_string)
// 名字不区分单复数，都用单数名词
#define N_NULL "null"
#define N_TRUE "true"
#define N_FALSE "false"
#define N_MS "ms"
#define N_CS "cs"
#define N_UPS "ups"
#define N_RS "rs"
#define N_TID "table_id"
#define N_REF_TID "ref_table_id"
#define N_DATA_TID "data_table_id"
#define N_CID "column_id"
#define N_COL_CONV_INFOS "column_conv_infos"
#define N_OLD_PROJECTOR "old_projector"
#define N_CID_SCAN "column_id_for_scan"
#define N_CID_UPDATE "column_id_for_update"
#define N_COLUMN_CONV_FUNCTION "column_conv"
#define N_PRIMARY_CID "primary_key_ids"
#define N_REFERED_CID "refered_ids"
#define N_DUP_CID "dup_column_id"
#define N_CELL_INDEX "cell_index"
#define N_HAS_INDEX "has_index"
#define N_COLUMN_INDEX "column_index"
#define N_COLUMN "column"
#define N_INSERT_COLUMN "insert_column"
#define N_UPDATE_COLUMN "update_column"
#define N_UPDATE_EXPR "update_expr"
#define N_PROJECT "project"
#define N_ASCENDING "ascending"
#define N_ASC "asc"
#define N_DESC "desc"
#define N_NULLS_FIRST_ASC "NULLs_first_asc"
#define N_NULLS_LAST_ASC "NULLs_last_asc"
#define N_NULLS_FIRST_DESC "NULLs_first_desc"
#define N_NULLS_LAST_DESC "NULLs_last_desc"
#define N_NULL_POS "null_position"
#define N_ORDER_BY "order_by"
#define N_TOPK "topk"
#define N_RANGE "range"
#define N_INDEX_RANGE "index_range"
#define N_LIMIT "limit"
#define N_WIN_FUNC "win_func"
#define N_OFFSET "offset"
#define N_ERROR_ON_OVERLAP_TIME "error_on_overlap_time"
#define N_LIMIT_OFFSET "limit_offset"
#define N_READ_METHOD "read_method"
#define N_SCAN "scan"
#define N_GET "get"
#define N_HAS_SCAN "has_scan"
#define N_HAS_GET "has_get"
#define N_IS_GET "is_get"
#define N_IS_STANDARD "is_standard"
#define N_ALIAS_TID "alias_table_id"
#define N_ROW_DESC "row_desc"
#define N_DESCRIPTOR "descriptor"
#define N_LOCK_FLAG "lock_flag"
#define N_WRITE "write"
#define N_NONE "none"
#define N_HOTSPOT "hotspot"
#define N_SUBQUERY_ID "subquery_id"
#define N_DML_TYPE "dml_type"
#define N_IS_UPS_ROW "is_ups_row"
#define N_DEFAULT_ROW "default_row"
#define N_DATA_SIZE "data_size"
#define N_BLOCK_NUM "block_num"
#define N_ROW "row"
#define N_ROW_COUNT "row_count"
#define N_EXPR "expression"
#define N_EXPLAIN_STMT "explain_stmt"
#define N_PARTITION_EXPR "partition_express"
#define N_SUBPARTITION_EXPR "partition_express"
#define N_IDX "idx"
#define N_CHILDREN_OPS "child_operators"
#define N_PHY_OP_ID "phy_operator_id"
#define N_CHILDREN_NUM "child_operators_num"
#define N_OP_NUM "operators_num"
#define N_SUBQUERY_NUM "subquery_num"
#define N_MAIN_QUERY "main_query"
#define N_SUBQUERIES "sub_queries"
#define N_TRANS_ID "tran_id"
#define N_START_TRANS "begin_trans"
#define N_START_TIME "start_time"
#define N_END_TIME "end_time"
#define N_EXPR_ID "expr_id"
#define N_IS_ALIAS "is_alias"
#define N_ALIAS_NAME "alias_name"
#define N_SYNONYM_NAME "synonym_name"
#define N_EXPR_NAME "expr_name"
#define N_STR_PART_NUM "part_num"
#define N_STR_PART_FUNC_TYPE "part_func_type"
#define N_STR_PART_EXPR_LEN "part_func_expr_len"
#define N_STR_PART_BIN_EXPR_LEN "part_func_bin_expr_len"
#define N_STR_PART_EXPR_TOTAL_LEN "part_fun_expr_total_len"
#define N_STR_PART_EXPR "part_func_expr"
#define N_STR_PART_BIN_EXPR "part_func_bin_expr"
#define N_OBJ_TYPE "obj_type"
#define N_RESULT_TYPE "result_type"
#define N_INPUT_TYPE "input_type"
#define N_CMP_TYPE "cmp_type"
#define N_JOINED_TID "joined_table_id"
#define N_JOIN_TYPE "join_type"
#define N_NUM "num"
#define N_SINH "sinh"
#define N_COSH "cosh"
#define N_DEGREES "degrees"
#define N_TANH "tanh"
#define N_RADIANS "radians"
#define N_IS_JOIN "is_join"
#define N_IS_MOCK "is_mock"
#define N_JOIN_EQ_COND "equal_join_conds"
#define N_JOIN_OTHER_COND "other_join_conds"
#define N_SIN "sin"
#define N_COS "cos"
#define N_TAN "tan"
#define N_COT "cot"
#define N_ASIN "asin"
#define N_ACOS "acos"
#define N_ATAN "atan"
#define N_ATAN2 "atan2"
#define N_PUMP_ROW_DESC "pump_row_desc"
#define N_ROOT_ROW_DESC "root_row_desc"
#define N_PSEUDO_COLUMN_ROW_DESC "pseudo_column_row_desc"
#define N_CONNECT_BY_PRIOR_EXPRS "connect_by_prior_exprs"
#define N_SUBSTMT "sub_stmt"
#define N_STMT_NAME "stmt_name"
#define N_PREPARE_SQL "prepare_sql"
#define N_DISTINCT "distinct"
#define N_ROLLUP "rollup"
#define N_NOCYCLE "nocycle"
#define N_FOR_UPDATE "for_update"
#define N_WAIT "wait"
#define N_MOCK_ID "mock_id"
#define N_MAX_WAIT_EVENT "max_wait_event"
#define N_SELECT "select"
#define N_INSERT_UPDATE "insert_up"
#define N_FROM "from"
#define N_JOINED_TABLE "joined_table"
#define N_SEMI_INFO "semi_info"
#define N_GROUP_BY_IDX "group_by_idx"
#define N_GROUP_BY "group_by"
#define N_ROLLUP_IDX "rollup_idx"
#define N_ROLLUP "rollup"
#define N_GROUPING_SETS "grouping sets"
#define N_AGG_PARAM_LIST "agg_param_list"
#define N_AGGR_COLUMN "aggr_col"
#define N_HAVING "having"
#define N_START_WITH "start_with"
#define N_CONNECT_BY "connect_by"
#define N_EXTRA_OUTPUT_EXPRS "extra_output_exprs"
#define N_AGGR_FUNC "aggr_func"
#define N_IS_SERVING_TENANT "is_serving_tenant"
#define N_SET_OP "set_op"
#define N_LEFT_QUERY "left_query"
#define N_RIGHT_QUERY "right_query"
#define N_ITEM_TYPE "item_type"
#define N_STMT_TYPE "stmt_type"
#define N_QID "query_id"
#define N_PLAN_ID "plan_id"
#define N_SQL_ID "sql_id"
#define N_CG_ID "column_group_id"
#define N_ROWKEY_ID "rowkey_id"
#define N_JOIN_TID "join_table_id"
#define N_JOIN_CID "join_column_id"
#define N_COLUMN_TYPE "column_type"
#define N_ENUM_SET_VALUES "enum_set_values"
#define N_LEN "len"
#define N_LENGTH "length"
#define N_LENGTHB "lengthb"
#define N_BIT_LENGTH "bit_length"
#define N_ZEROFILL "zf"
#define N_VALUES "values"
#define N_ROWKEY_TO_ROWID "rowkey_to_rowid"
#define N_PRECISION "precision"
#define N_SCALE "scale"
#define N_NULLABLE "nullable"
#define N_AUTO_FILL_TIMESTAMP "auto_filled_timestamp"
#define N_EXISTS "exists"
#define N_NOT_EXISTS "not exists"
#define N_NUMBER_PRECISION "number_precision"
#define N_NUMBER_SCALE "number_scale"
#define N_PARAM "param"
#define N_PARAM_NUM "param_num"
#define N_RESCAN_PARAM "rescan_param"
#define N_ONETIME_FILTER "onetime_filter"
#define N_FETCH_CUR_TIME "fetch_cur_time"
#define N_SYS_VAR "sys_var"
#define N_SYS_VAR_SCOPE "sys_var_scope"
#define N_USER_VAR "user_var"
#define N_OP "op"
#define N_FUNC "func"
#define N_UDF "user_define_function"
#define N_CONST "const"
#define N_POST_EXPR "post_expr"
#define N_TIMEOUT "timeout"
#define N_CACHE_BLOOM_FILTER "cache_bloom"
#define N_CACHE_FROZEN_DATA "cache_frozen_data"
#define N_READ_CONSISTENCY "read_consistency"
#define N_UPS_SCAN_TYPE "ups_scan_type"
#define N_IS_CONSITENCY "consistency_read"
#define N_ONLY_STATIC_DATA "only_static"
#define N_ONLY_FROZEN_VERSION "only_frozen_version_data"
#define N_VERSION_RANGE "version_range"
#define N_FROZEN_VERSION "frozen_version"
#define N_VERSION "version"
#define N_OB_VERSION "ob_version"
#define N_ICU_VERSION "icu_version"
#define N_CONNECTION_ID "connection_id"
#define N_SESSIONTIMEZONE "sessiontimezone"
#define N_DBTIMEZONE "dbtimezone"
#define N_SYS_EXTRACT_UTC "sys_extract_utc"
#define N_TZ_OFFSET "tz_offset"
#define N_FROM_TZ "from_tz"
#define N_SCAN_PARAM "scan_param"
#define N_SCAN_FLAG "scan_flag"
#define N_GET_PARAM "get_param"
#define N_ROWKEY_LIST "rowkey_list"
#define N_READ_MASTER "read_master"
#define N_RESULT_CACHED "result_cached"
#define N_NOT_EXIST_COL_RET_NOP "not_exsit_col_ret_as_nop"
#define N_RAW_EXPR "raw_expr"
#define N_ROWKEY_CELL_NUM "rowkey_cell_num"
#define N_CELL "cell"
#define N_RESERVED_CELL_COUNT "reserved_cell_count"
#define N_SHARDING "shardings"
#define N_MAP_PLAN "map_plan"
#define N_REDUCE_PLAN "reduce_plan"
#define N_PLAN "plan"
#define N_PLAN_TREE "plan_tree"
#define N_INDEX "index"
#define N_INDEX_SHORT_NAME "index_short_name"
#define N_SORT_COLUMN "sort_columns"
#define N_INDEX_COVER_PREFIX "cover_prefix"
#define N_INDEX_TABLE "index_table"
#define N_INDEX_TABLE_SIZE "index_table_size"
#define N_INDEX_COL_SIZE "index_column_size"
#define N_INDEX_ROW_SIZE "index_row_size"
#define N_INDEX_TID "index_table_id"
#define N_INDEX_ID "index_id"
#define N_ORDERBY_ELI "orderby_eliminate"
#define N_GROUPSORT_ELI "group_sort_eliminate"
#define N_IS_PRIMARY_INDEX "is_primary_index"
#define N_ALL_COVERED "all_covered"
#define N_ALTER_TABLE_SCHEMA "alter_table_schema"
#define N_MASTER_IP "master_ip"
#define N_PORT "port"
#define N_IS_FORCE "is_force"
#define N_VARIABLE_NAME "var_name"
#define N_VARIABLE_VALUE "var_value"
#define N_VARIABLE "var"
#define N_VERBOSE "verbose"
#define N_SHOW_STMT_CTX "show_stmt_ctx"
#define N_TABLE "table"
#define N_TABLE_ID "table_id"
#define N_SHOW_TABLE_ID "show_table_id"
#define N_BASE_TABLE_ID "base_table_id"
#define N_TABLE_IDS "table_ids"
#define N_IF_EXISTS "if_exists"
#define N_TABLE_NAME "table_name"
#define N_DATABASE_NAME "database_name"
#define N_DBLINK_ID "dblink_id"
#define N_DBLINK_NAME "dblink_name"
#define N_LINK_TABLE_ID "link_table_id"
#define N_LINK_TABLE_NAME "link_table_name"
#define N_QB_NAME "qb_name"
#define N_AUTOINC_NEXTVAL "nextval"
#define N_TABLET_AUTOINC_NEXTVAL "tablet_autoinc"
#define N_SEQ_NEXTVAL "seq_value"
#define N_RESERVED_CELL_COUNT "reserved_cell_count"
#define N_COLUMN_COUNT "column_count"
#define N_COLUMN_ID "column_id"
#define N_COLUMN_IDS "column_ids"
#define N_ALTER_TYPE "alter_type"
#define N_ROLLBACK "rollback"
#define N_WHEN_NUMBER "when_number"
#define N_IS_KILL_QUERY "is_kill_query"
#define N_SESSION_ID "session_id"
#define N_ROW_INTERVAL "row_interval"
#define N_IS_DELETE "is_delete"
#define N_ROWSTORE "rowstore"
#define N_KEY "key"
#define N_PG_KEY "pg_key"
#define N_KEY_RANGES "key_ranges"
#define N_VALUE "value"
#define N_CONTENT "content"
#define N_ROOT "root"
#define N_HUSK_SCAN "husk_scan"
#define N_JOIN_TYPE "join_type"
#define N_SNAPSHOT_TS "snapshot_ts"
#define N_EXPIRE_CONDITION "expire_condition"
#define N_STR_LEN "str_len"
#define N_SQL_EXPR_LEN "sql_expr_len"
#define N_TOTAL_LEN "total_len"
#define N_START_TRANS_FLAG "start_trans_flag"
#define N_SPECIFY_INDEX "specify_index"
#define N_SESSION_TIMEOUT_TS "session_timeout"
#define N_SESSION_IDLE_TIMEOUT_TS "session_idle_timeout"
#define N_QUERY_RANGE "query_range"
#define N_RANGE_GRAPH "range_graph"
#define N_KEY_PART_VAL "key_part_value"
#define N_ITEM_KEY_PART "item_key_part"
#define N_AND_KEY_PART "and_key_part"
#define N_OR_KEY_PART "or_key_part"
#define N_START_VAL "start_value"
#define N_END_VAL "end_value"
#define N_INCLUDE_START "include_start"
#define N_INCLUDE_END "include_end"
#define N_IS_STRICT_IN "is_strict_in"
#define N_CONTAIN_QUESTIONMARK "contain_questionmark"
#define N_OFFSETS "offsets"
#define N_MISSING_OFFSETS "missing_offsets"
#define N_IN_PARAMS "in_params"
#define N_PATTERN_VAL "pattern"
#define N_ESCAPE_VAL "escape"
#define N_ALWAYS_TRUE "always_true"
#define N_ALWAYS_FALSE "always_false"
#define N_RPC_ROW_ITER "rpc_row_iter"
#define N_ABS_EXPIRED_TIME "abs_expired_time"
#define N_VIEW_DEFINITION "view_definition"
#define N_CHECK_OPTION "check_option"
#define N_IS_UPDATABLE "is_updatable"
#define N_IS_MATERIALIZED "is_materialized"
#define N_VIEW_SCHEMA "view_schema"
#define N_FILTER_EXPRS "filter_exprs"
#define N_VIRTUAL_COLUMN_EXPRS "virtual_column_exprs"
#define N_CALC_EXPRS "calc_exprs"
#define N_INDEX_FILTER_EXPRS "index_filter_exprs"
#define N_OPERATOR_MONITOR_INFO "op_info"
#define N_PLAN_MONITOR_INFO "plan_info"
#define N_ANY_VAL "any_value"
#define N_VALIDATE_PASSWORD_STRENGTH "validate_password_strength"
#define N_ENCODE_SORTKEY "encode_sortkey"
#define N_HASH "hash"
#define N_SHA "sha"
#define N_SHA1 "sha1"
#define N_SHA2 "sha2"
#define N_COMPRESS "compress"
#define N_UNCOMPRESS "uncompress"
#define N_UNCOMPRESSED_LENGTH "uncompressed_length"
#define N_STATEMENT_DIGEST "statement_digest"
#define N_STATEMENT_DIGEST_TEXT "statement_digest_text"
//common comparison operator
#define N_LESS_THAN "<"
#define N_GREATER_THAN ">"
#define N_EQUAL "="
#define N_NS_EQUAL "<=>"
#define N_LESS_EQUAL "<="
#define N_GREATER_EQUAL ">="
#define N_NOT_EQUAL "!="
#define N_IS "is"
#define N_IS_NOT "is_not"
#define N_BTW "between"
#define N_NOT_BTW "not_between"
//subquery comparison operator
#define N_SQ_LESS_THAN "subquery_less_than"
#define N_SQ_GREATER_THAN "subquery_greater_than"
#define N_SQ_EQUAL "subquery_equal"
#define N_SQ_NS_EQUAL "subquery_null_safe_equal"
#define N_SQ_LESS_EQUAL "subquery_less_equal"
#define N_SQ_GREATER_EQUAL "subquery_greater_equal"
#define N_SQ_NOT_EQUAL "subquery_not_equal"

#define N_REMOVE_CONST "remove_const"
#define N_WRAPPER_INNER "wrapper_inner"


#define N_COLUMN_REF "column_ref"
#define N_NEG "neg"
#define N_PRIOR "prior"
#define N_ABS "abs"
#define N_ADD "+"
#define N_MINUS "-"
#define N_MUL "*"
#define N_DIV "/"
#define N_MOD "%"
#define N_AND "&&"
#define N_OR "||"
#define N_NOT "!"
#define N_POW "pow"
#define N_XOR "^"
#define N_ROWEQ "row_eq"
#define N_ROWLE "row_le"
#define N_ROWLT "row_lt"
#define N_ROWGE "row_ge"
#define N_ROWGT "row_gt"
#define N_ROWNEQ "row_neq"
#define N_IN "in"
#define N_NOT_IN "not_in"
#define N_INT_DIV "div"
#define N_REGEXP "regexp"
#define N_NOT_REGEXP "not_regexp"
#define N_REGEXP_SUBSTR "regexp_substr"
#define N_REGEXP_INSTR "regexp_instr"
#define N_REGEXP_REPLACE "regexp_replace"
#define N_REGEXP_COUNT "regexp_count"
#define N_REGEXP_LIKE "regexp_like"
#define N_LIKE "like"
#define N_NOT_LIKE "not_like"
#define N_SUBSTR "substr"
#define N_INITCAP "initcap"
#define N_MID "mid"
#define N_SUBSTRB "substrb"
#define N_STRCMP "strcmp"
#define N_INSERT "insert"
#define N_SUBSTRING_INDEX "substring_index"
#define N_MD5 "md5"
#define N_CRC32 "crc32"
#define N_HEX "hex"
#define N_UNHEX "unhex"
#define N_HEXTORAW "hextoraw"
#define N_RAWTOHEX "rawtohex"
#define N_RAWTONHEX "rawtonhex"
#define N_UTL_RAW_CAST_TO_RAW "utl_raw_cast_to_raw"
#define N_UTL_RAW_CAST_TO_VARCHAR2 "utl_raw_cast_to_varchar2"
#define N_UTL_RAW_LENGTH "utl_raw_length"
#define N_UTL_RAW_BIT_AND "utl_raw_bit_and"
#define N_UTL_RAW_BIT_OR "utl_raw_bit_or"
#define N_UTL_RAW_BIT_XOR "utl_raw_bit_xor"
#define N_UTL_RAW_BIT_COMPLEMENT "utl_raw_bit_complement"
#define N_UTL_RAW_REVERSE "utl_raw_reverse"
#define N_UTL_RAW_COPIES "utl_raw_copies"
#define N_UTL_RAW_COMPARE "utl_raw_compare"
#define N_UTL_RAW_SUBSTR "utl_raw_substr"
#define N_UTL_RAW_CONCAT "utl_raw_concat"
#define N_UTL_I18N_STRING_TO_RAW "utl_i18n_string_to_raw"
#define N_UTL_I18N_RAW_TO_CHAR "utl_i18n_raw_to_char"
#define N_UTL_INADDR_GET_HOST_ADDR "utl_inaddr_get_host_address"
#define N_UTL_INADDR_GET_HOST_NAME "utl_inaddr_get_host_name"
#define N_DBMS_LOB_GETLENGTH "dbms_lob_getlength"
#define N_DBMS_LOB_APPEND "dbms_lob_append"
#define N_DBMS_LOB_READ "dbms_lob_read"
#define N_DBMS_LOB_CONVERTTOBLOB "dbms_lob_converttoblob"
#define N_DBMS_LOB_CAST_CLOB_TO_BLOB "dbms_lob_cast_clob_to_blob"
#define N_DBMS_LOB_CONVERT_CLOB_CHARSET "dbms_lob_convert_clob_charset"
#define N_IP2INT "ip2int"
#define N_INT2IP "int2ip"
#define N_INETATON "inet_aton"
#define N_INET6ATON "inet6_aton"
#define N_INET6NTOA "inet6_ntoa"
#define N_IS_IPV4 "is_ipv4"
#define N_IS_IPV6 "is_ipv6"
#define N_IS_IPV4_MAPPED "is_ipv4_mapped"
#define N_IS_IPV4_COMPAT "is_ipv4_compat"
#define N_UPPER "upper"
#define N_SIGN "sign"
#define N_LOWER "lower"
#define N_NLS_LOWER "nls_lower"
#define N_NLS_UPPER "nls_upper"
#define N_REPEAT "repeat"
#define N_REPLACE "replace"
#define N_TRANSLATE "translate"
#define N_CONCAT "concat"
#define N_EXPORT_SET "export_set"
#define N_CONCAT_WS "concat_ws"
#define N_TO_OUTFILE_ROW "to_outfile_row"
#define N_INSTR "instr"
#define N_INSTRB "instrb"
#define N_CONV "conv"
#define N_SYS_VIEW_BIGINT_PARAM "sys_view_bigint_param"
#define N_SYS_PRIVILEGE_CHECK "sys_privilege_check"
#define N_WIDTH_BUCKET "width_bucket"
#define N_LOCATE "locate"
#define N_POSITION "position"
#define N_BIN "bin"
#define N_QUOTE "quote"
#define N_TRIM "trim"
#define N_INNER_TRIM "inner_trim"
#define N_PART_HASH "partition_hash"
#define N_PART_KEY "partition_key"
#define N_ADDR_TO_PARTITION_ID "addr_to_partition_id"
#define N_CAST "cast"
#define N_TREAT "treat"
#define N_REMAINDER "remainder"
#define N_TO_TYPE "to_type"
#define N_TO_NUMBER "to_number"
#define N_CHAR "char"
#define N_CONVERT "convert"
#define N_GREATEST "greatest"
#define N_LEAST "least"
#define N_COALESCE "coalesce"
#define N_NVL "nvl"
#define N_NVL2 "nvl2"
#define N_FORMAT_BYTES "format_bytes"
#define N_FORMAT_PICO_TIME "format_pico_time"
#define N_EXPR_TYPE "expr_type"
#define N_DIM "dimension"
#define N_CASE "case"
#define N_DECODE "decode"
#define N_ARG_CASE "arg_case"
#define N_SCAN_PLAN "scan_plan"
#define N_GET_PLAN "get_plan"
#define N_TABLE_TYPE "table_type"
#define N_REF_ID "ref_id"
#define N_REF_QUERY "ref_query"
#define N_OBJ_ACCESS "obj_access"
#define N_MULTISET "multiset"
#define N_LEFT_TABLE "left_table"
#define N_RIGHT_TABLE "right_table"
#define N_COLUMN_NAME "column_name"
#define N_WHEN "when"
#define N_WHERE "where"
#define N_MONTH "month"
#define N_MONTH_NAME "monthname"
#define N_DATE "date"
#define N_DATE_ADD "date_add"
#define N_DATE_SUB "date_sub"
#define N_DATE_DIFF "datediff"
#define N_TIME_STAMP_DIFF "timestampdiff"
#define N_TIME_DIFF "timediff"
#define N_PERIOD_DIFF "period_diff"
#define N_PERIOD_ADD "period_add"
#define N_SYS_SLEEP "sleep"
#define N_SYS_ERRNO "errno"
#define N_UNIX_TIMESTAMP "unix_timestamp"
#define N_FROM_UNIX_TIME "from_unixtime"
#define N_EXTRACT "extract"
#define N_DATE_FORMAT "date_format"
#define N_STR_TO_DATE "str_to_date"
#define N_TO_DATE "to_date"
#define N_TO_CHAR "to_char"
#define N_TIMESTAMP "timestamp"
#define N_MAKEDATE "makedate"
#define N_GET_FORMAT "get_format"
#define N_FORMAT "format"
#define N_TO_CLOB "to_clob"
#define N_TO_BLOB "to_blob"
#define N_EMPTY_CLOB "empty_clob"
#define N_EMPTY_BLOB "empty_blob"
#define N_TO_TIMESTAMP "to_timestamp"
#define N_TO_TIMESTAMP_TZ "to_timestamp_tz"
#define N_TO_DAYS "to_days"
#define N_DAY_OF_MONTH "dayofmonth"
#define N_DAY "day"
#define N_DAY_OF_WEEK "dayofweek"
#define N_DAY_OF_YEAR "dayofyear"
#define N_DAY_NAME "dayname"
#define N_HOUR "hour"
#define N_SECOND "second"
#define N_MINUTE "minute"
#define N_MICROSECOND "microsecond"
#define N_TO_SECONDS "to_seconds"
#define N_TIME_TO_SEC "time_to_sec"
#define N_SEC_TO_TIME "sec_to_time"
#define N_SUB_TIME "subtime"
#define N_ADD_TIME "addtime"
#define N_FROM_DAYS "from_days"
#define N_DATABASE "database"
#define N_TRACE_ID "trace_id"
#define N_LAST_TRACE_ID "last_trace_id"
#define N_ROW_COUNT "row_count"
#define N_FOUND_ROWS "found_rows"
#define N_LAST_INSERT_ID "last_insert_id"
#define N_LAST_INSERT_ID_TO_CLIENT "last_insert_id_to_client"
#define N_LAST_INSERT_ID_SESSION "last_insert_id_session"
#define N_SYSDATE "sysdate"
#define N_CUR_TIMESTAMP "current_timestamp"
#define N_LOCALTIMESTAMP "localtimestamp"
#define N_TIMESTAMP_NVL "timestamp_nvl"
#define N_CUR_TIME "curtime"
#define N_CUR_DATE "cur_date"
#define N_CURRENT_DATE "current_date"
#define N_UTC_TIMESTAMP "utc_timestamp"
#define N_UTC_TIME "utc_time"
#define N_UTC_DATE "utc_date"
#define N_SYSTIMESTAMP "systimestamp"
#define N_MAKETIME "maketime"
#define N_UPS_CUR_TIME "ups_cur_time"
#define N_REAL_UPS_TIME "real_ups_time"
#define N_MERGING_FROZEN_TIME "merging_frozen_time"
#define N_POS "pos"
#define N_SCHEMA_VERSION "schema_version"
#define N_QUERY_BEGIN_SCHEMA_VERSION "query_begin_schema_version"
#define N_PROGRESSIVE_MERGE_NUM "progressive_merge_num"
#define N_TABLE_SCHEMA_VERSION "table_schema_version"
#define N_QUESTION_MARK "?"
#define N_FIRST "first"
#define N_SECOND "second"
#define N_ZONE "zone"
#define N_EXPR_INFO "expr_info"
#define N_REL_ID "rel_id"
#define N_BASE_COLUMN_ID "base_column_id"
#define N_ALIAS_COLUMN_ID "alias_column_id"
#define N_REF_HANDLE "ref_handle"
#define N_COMMENT "comment"
#define N_ID "id"
#define N_SOUNDEX "soundex"
#define N_SCOPE "scope"
#define N_TYPE "type"
#define N_INT_VALUE "int_val"
#define N_STR_VALUE "str_val"
#define N_STR_VALUE_LEN "str_len"
#define N_CHILDREN "children"
#define N_SEPARATOR_PARAM_EXPR "separator_param_expr"
#define N_LINEAR_INTER_EXPR "linear_inter_expr"
#define N_TIME_TO_USEC "time_to_usec"
#define N_USEC_TO_TIME "usec_to_time"
#define N_CONSISTENCY_LEVEL "consistency_level"
#define N_REPLICA_NUM "replica_num"
#define N_TABLET_MAX_SIZE "tablet_max_size"
#define N_TABLET_BLOCK_SIZE "tablet_block_size"
#define N_INDEX_STATUS "index_status"
#define N_ALTER_TABLE_OPTION "alter_table_option"
#define N_ALTER_COLUMN_SCHEMA "alter_column_schema"
#define N_CHARSET_TYPE "charset_type"
#define N_COLLATION_TYPE "collation_type"
#define N_USE_BLOOM_FILTER "use_bloom_filter"
#define N_COMPRESS_METHOD "compress_method"
#define N_JOIN_INFO "join_info"
#define N_ID_SET "id_set"
#define N_EQUAL_SET "equal_set"
#define N_LEFT_ID "left_id"
#define N_RIGHT_ID "right_id"
#define N_LEFT_NAME "left_name"
#define N_RIGHT_NAME "right_name"
#define N_OUTER_ID "outer_id"
#define N_INNER_ID "inner_id"
#define N_OUTER "outer"
#define N_INNER "inner"
#define N_SESSION_KEY "session_key"
#define N_SESSION_IDX "session_idx"
#define N_SESSION_REUSE_COUNT "session_reuse_count"
#define N_FIRST_ROWKEY "first_rowkey"
#define N_KEEP_TRANS "keep_trans"
#define N_NEED_REFRESH_SNAPSHOT "need_refresh_snapshot"
#define N_TRANS_REQ "trans_req"
#define N_TRANS_DESC "trans_desc"
#define N_IS_UPDATE_ROWKEY "is_update_rowkey"
#define N_FLOW_PERCENT "flow_percent"
#define N_START_LSN "start_lsn"
#define N_BUFFER "buf"
#define N_LSN "lsn"
#define N_TS "ts"
#define N_PROPOSE_ID "propose_id"
#define N_CHECKSUM "checksum"
#define N_PARTITION_IDX "partition_idx"
#define N_PARTITION_CNT "partition_cnt"
#define N_OCT "oct"
#define N_RPAD "rpad"
#define N_ASSIGN "assign"
#define N_GET_USER_VAR "get_user_var"
#define N_GET_SYS_VAR "get_sys_var"
#define N_GET_PACKAGE_VAR "get_package_var"
#define N_GET_SUBPROGRAM_VAR "get_subprogram_var"
#define N_SHADOW_UK_PROJECTOR "shadow_uk_project"
#define N_RANDOM_BYTES "random_bytes"
//use capital as 'default now()' flag
#define N_UPPERCASE_CUR_TIMESTAMP "CURRENT_TIMESTAMP"
#define N_UPDATE_CURRENT_TIMESTAMP "ON UPDATE CURRENT_TIMESTAMP"
#define N_UPS_ADDR "ups_addr"
#define N_ADD_ADDR_LIST "add_addr_list"
#define N_DELETE_ADDR_LIST "delete_addr_list"
#define N_UPS_COUNT_LIMIT "ups_count_limit"
#define N_HUSK_TABLET_BASE "husk_tablet_base"
#define N_DECIMAL_SCALE "decimal_scale"
#define N_READ_ONLY "read_only"
#define N_BLOCK_SIZE "block_size"
#define N_AFFECTED_ROWS "affected_rows"
#define N_ERR_CODE "err_code"
#define N_ERR_MSG "err_msg"
#define N_MEM_LIMIT "mem_limit"
#define N_ROW "row"
#define N_ROW_STORE "row_store"
#define N_PAYLOAD "payload"
#define N_CELLS "cells"
#define N_OB_EXECUTION_ID "ob_execution_id"
#define N_TIME_FORMAT "time_format"
#define N_EXECUTION_TIME "execution_time"
#define N_OB_JOB_ID "ob_job_id"
#define N_OB_TASK_ID "ob_task_id"
#define N_OB_SLICE_ID "ob_slice_id"
#define N_EXECUTION_ID "execution_id"
#define N_TASK_TYPE "task_type"
#define N_LAST_EXECUTION_ID "last_execution_id"
#define N_JOB_ID "job_id"
#define N_TASK_ID "task_id"
#define N_OP_ID "op_id"
#define N_SLICE_ID "slice_id"
#define N_PARENT_EXEC_COND "parent_exec_cond"
#define N_JOB_ATTR "job_attr"
#define N_JOB_COUNT "job_count"
#define N_TASK_COUNT "task_count"
#define N_SLICE_COUNT "slice_count"
#define N_ROOT_JOB "root_job"
#define N_CHILD_JOB "child_job"
#define N_JOB "job"
#define N_JOB_TREE "job_tree"
#define N_TASK_INFO "task_info"
#define N_SERVER "server"
#define N_SERVER_ADDR "server_addr"
#define N_TASK_LOC "task_loc"
#define N_RANGE_LOC "range_loc"
#define N_TASK_STATE "task_state"
#define N_THEN "then"
#define N_PAD "pad"
#define N_COLUMN_CONV "column_conv"
#define N_DEFAULT "default"
#define N_ORA_DECODE "ora_decode"
#define N_ORA_TRUNC "ora_trunc"
#define N_TRUNC "trunc"
#define N_ROUND "round"
#define N_TO_BINARY_FLOAT "TO_BINARY_FLOAT"
#define N_TO_BINARY_DOUBLE "TO_BINARY_DOUBLE"
#define T_NANVL "NANVL"
#define N_RANGE_COLUMNS "range_columns"
#define N_START_KEY "startkey"
#define N_END_KEY "endkey"
#define N_BORDER "border"
#define N_FLAG "flag"
#define N_FLAGS "flags"
#define N_MAX_LENGTH "max_length"
#define N_CALC_TYPE "calc_type"
#define N_CALC_META "calc_meta"
#define N_AST "AST"
#define N_UPDATED_CID "updated_column_id"
#define N_REAL_PARAM_NUM "real_param_num"
#define N_PROJECTOR "projector"
#define N_CMD_TYPE "cmd_type"
#define N_JOIN_ORDER "join_order"
#define N_STMT_HINT "hint"
#define N_SUBQUERY_EXPRS "subquery_exprs"
#define N_USER_VARS "user_variables"
#define N_QUERY_CTX "query_context"
#define N_UPPER_BASE_COLUMNS "upper_base_columns"
#define N_UPPER_ALIAS_COLUMNS "upper_alias_columns"
#define N_GROUP_COLUMN_REF_LEVELS "group_column_ref_levels"
#define N_NON_GROUP_COLUMN_REF_LEVELS "non_group_column_ref_levels"
#define N_CHILD_STMT "child_stmt"
#define N_PURE_TABLE_ID "pure_table_id"
#define N_TENANT_ID "tenant_id"
#define N_ON_DUPLICATE "on_duplicate"
#define N_CREATE_TABLE_ARG "create_table_arg"
#define N_DEFAULT_VALUE "default_value"
#define N_TENANT "tenant"
#define N_EFFECTIVE_TENANT "effective_tenant"
#define N_EFFECTIVE_TENANT_ID "effective_tenant_id"
#define N_CURRENT_USER "current_user"
#define N_CURRENT_USER_PRIV "current_user_priv"
#define N_USER "user"
#define N_HOST_IP "host_ip"
#define N_RPC_PORT "rpc_port"
#define N_MYSQL_PORT "mysql_port"
#define N_CHARSET "charset"
#define N_COLLATION "collation"
#define N_COERCIBILITY "coercibility"
#define N_SET_COLLATION "set_collation"
#define N_META "meta"
#define N_OBJ "obj"
#define N_ACCURACY "accuracy"
#define N_CHAR_LENGTH "char_length"
#define N_BIT_AND "&"
#define N_BIT_AND_ORACLE "BITAND"
#define N_BIT_OR "|"
#define N_BIT_XOR "^"
#define N_BIT_NEG "~"
#define N_BIT_LEFT_SHIFT "<<"
#define N_BIT_RIGHT_SHIFT ">>"
#define N_IFNULL "ifnull"
#define N_ALIAS_REF "alias_ref"
#define N_INDEX_ARG "index_arg"
#define N_FOREIGN_KEY_ARG "foreign_key_arg"
#define N_FIELD "field"
#define N_ELT "elt"
#define N_NULLIF "nullif"
#define N_IS_TOTAL_QUANTITY_LOG "is_total_quantity_log"
#define N_ISNULL "isnull"
#define N_SQL_MODE "sql_mode"
#define N_IS_IGNORE "is_ignore"

#define N_PRIMARY "primary"
#define N_PART_ID "part_id"
#define N_INNER_GET "inner_get"
#define N_MATCH_AGAINST "match_against"
#define N_WORD_SEGMENT "word_segment"
#define N_SELF_JOIN "self_join"
#define N_DES_HEX_STR "DES_HEX_STR"
#define N_YEAR "year"
#define N_TIME "time"
#define N_UUID "uuid"
#define N_UUID_SHORT "uuid_short"
#define N_SYS_GUID "sys_guid"
#define N_UUID_TO_BIN "uuid_to_bin"
#define N_IS_UUID "is_uuid"
#define N_BIN_TO_UUID "bin_to_uuid"
#define N_SET_TO_STR "set_to_str"
#define N_ENUM_TO_STR "enum_to_str"
#define N_SET_TO_INNER_TYPE "set_to_inner_type"
#define N_ENUM_TO_INNER_TYPE "enum_to_inner_type"
#define N_ASCII "ascii"
#define N_ORD "ord"
#define N_LTRIM "ltrim"
#define N_RTRIM "rtrim"
#define N_LPAD "lpad"
#define N_SPACE "space"
#define N_TRUNCATE "truncate"
#define N_PI "pi"
#define N_TIME_STAMP_ADD "timestampadd"
#define N_CONNECT_BY_ROOT "connect_by_root"
#define N_SYS_CONNECT_BY_PATH "sys_connect_by_path"
#define N_CHR "chr"
#define N_EXP "exp"
#define N_CALC_UROWID "calc_urowid"
#define N_NAME_CONST "name_const"
//for dll udf
#define N_NORMAL_UDF "dll_normal_user_defined_function"
#define N_AGG_UDF "dll_agg_user_defined_function"

#define N_SQRT "sqrt"
#define N_LOG2 "log2"
#define N_LOG10 "log10"

#define N_WEEK_OF_YEAR "weekofyear"
#define N_WEEKDAY_OF_DATE "weekday"
#define N_YEARWEEK_OF_DATE "yearweek"
#define N_WEEK "week"
#define N_QUARTER "quarter"
#define N_USERENV "USERENV"
#define N_SYS_CONTEXT "SYS_CONTEXT"
#define N_SYS_CONTEXT "SYS_CONTEXT"
#define N_TIMESTAMP_TO_SCN "TIMESTAMP_TO_SCN"
#define N_SCN_TO_TIMESTAMP "SCN_TO_TIMESTAMP"

#define N_AES_ENCRYPT "aes_encrypt"
#define N_AES_DECRYPT "aes_decrypt"
#define N_DECODE "decode"
#define N_ENCODE "encode"
#define N_DES_DECRYPT "des_decrypt"
#define N_DES_ENCRYPT "des_encrypt"
#define N_ENCRYPT "encrypt"


#define N_UID "uid"
#define N_PL_INTEGER_CHECKER "pl_integer_checker"
#define N_PL_GET_CURSOR_ATTR "pl_get_cursor_attr"
#define N_PL_ASSOCIATIVE_INDEX "pl_associative_index"
#define N_PL_GET_SQLCODE_SQLERRM "pl_get_sqlcode_sqlerrm"
#define N_PLSQL_VARIABLE "plsql_variable"
#define N_PL_COLLECTION_CONSTRUCT "pl_collection_construct"
#define N_PL_SUBQUERY_CONSTRUCT "pl_subquery_construct"
#define N_PL_OBJECT_CONSTRUCT "pl_object_construct"

#define N_OUTER_JOIN_SYMBOL "(+)"

//for Lable Security

#define N_OLS_POLICY_CREATE               "ols_policy_create"
#define N_OLS_POLICY_ALTER                "ols_policy_alter"
#define N_OLS_POLICY_DROP                 "ols_policy_drop"
#define N_OLS_POLICY_DISABLE              "ols_policy_disable"
#define N_OLS_POLICY_ENABLE               "ols_policy_enable"

#define N_OLS_LEVEL_CREATE                "ols_level_create"
#define N_OLS_LEVEL_ALTER                 "ols_level_alter"
#define N_OLS_LEVEL_DROP                  "ols_level_drop"

#define N_OLS_COMPARMENT_CREATE           "ols_compartment_create"
#define N_OLS_COMPARMENT_ALTER            "ols_compartment_alter"
#define N_OLS_COMPARMENT_DROP             "ols_compartment_drop"

#define N_OLS_GROUP_CREATE                "ols_group_create"
#define N_OLS_GROUP_ALTER                 "ols_group_alter"
#define N_OLS_GROUP_DROP                  "ols_group_drop"

#define N_OLS_LABEL_CREATE                "ols_label_create"
#define N_OLS_LABEL_ALTER                 "ols_label_alter"
#define N_OLS_LABEL_DROP                  "ols_label_drop"

#define N_OLS_TABLE_POLICY_APPLY          "ols_table_policy_apply"
#define N_OLS_TABLE_POLICY_REMOVE         "ols_table_policy_remove"
#define N_OLS_TABLE_POLICY_DISABLE        "ols_table_policy_disable"
#define N_OLS_TABLE_POLICY_ENABLE         "ols_table_policy_enable"

#define N_OLS_SCHEMA_POLICY_APPLY         "ols_schema_policy_apply"
#define N_OLS_SCHEMA_POLICY_REMOVE        "ols_schema_policy_remove"
#define N_OLS_SCHEMA_POLICY_DISABLE       "ols_schema_policy_disable"
#define N_OLS_SCHEMA_POLICY_ENABLE        "ols_schema_policy_enable"

#define N_OLS_USER_SET_LEVELS             "ols_user_set_levels"
#define N_OLS_USER_SET_COMPARTMENTS       "ols_user_set_compartments"
#define N_OLS_USER_SET_GROUPS             "ols_user_set_groups"
#define N_OLS_USER_ADD_COMPARTMENTS       "ols_user_add_compartments"
#define N_OLS_USER_ALTER_COMPARTMENTS     "ols_user_alter_compartments"
#define N_OLS_USER_DROP_COMPARTMENTS      "ols_user_drop_compartments"
#define N_OLS_USER_DROP_ALL_COMPARTMENTS  "ols_user_drop_all_compartments"

#define N_OLS_LABEL_VALUE_CMP_LE          "ols_label_value_cmp_le"
#define N_OLS_LABEL_VALUE_CHECK           "ols_label_value_check"
#define N_OLS_LABEL_VALUE_TO_CHAR         "label_to_char"
#define N_OLS_CHAR_TO_LABEL_VALUE         "char_to_label"

#define N_OLS_SESSION_SET_LABEL               "ols_session_set_label"
#define N_OLS_SESSION_SET_ROW_LABEL           "ols_session_set_row_label"
#define N_OLS_SESSION_RESTORE_DEFAULT_LABEL   "ols_session_restore_default_label"
#define N_OLS_SESSION_SAVE_DEFAULT_LABEL      "ols_session_savle_default_label"

#define N_OLS_SESSION_LABEL                   "ols_session_label"
#define N_OLS_SESSION_ROW_LABEL               "ols_session_row_label"

#define N_VSIZE                             "vsize"
#define N_ORAHASH                           "ora_hash"

#define N_ADD_MONTHS                        "add_months"
#define N_LAST_DAY                          "last_day"
#define N_MONTHS_BETWEEN                    "months_between"
#define N_NEXT_DAY                          "next_day"
#define N_TO_DSINTERVAL                     "to_dsinterval"
#define N_TO_YMINTERVAL                     "to_yminterval"
#define N_NUMTODSINTERVAL                   "numtodsinterval"
#define N_NUMTOYMINTERVAL                   "numtoyminterval"

#define N_POWER                             "power"
#define N_LN                                "ln"
#define N_LOG                               "log"

#define N_PL_SEQ_NEXTVAL                    "pl_seq_nextval"
#define N_DUMP                              "dump"
#define N_BOOL                              "bool"
#define N_CALC_PARTITION_ID                 "calc_partition_id"
#define N_CALC_TABLET_ID                    "calc_tablet_id"
#define N_CALC_PARTITION_TABLET_ID          "calc_partition_tablet_id"
#define N_PDML_PARTITION_ID                 "pdml_partition_id"

#define N_TO_SINGLE_BYTE                    "to_single_byte"
#define N_TO_MULTI_BYTE                     "to_multi_byte"

#define N_DBMS_CRYPTO_ENCRYPT               "dbms_crypto_encrypt"
#define N_DBMS_CRYPTO_DECRYPT               "dbms_crypto_decrypt"

#define N_TO_NCHAR                          "to_nchar"
#define N_LNNVL                             "lnnvl"
#define N_SET                               "set"
#define N_CARDINALITY                       "cardinality"
#define N_COLL_PRED                         "coll_pred"
#define N_USER_CAN_ACCESS_OBJ               "user_can_access_obj"
#define N_IS_MULTI_TABLE_INSERT             "is_multi_table_insert"
#define N_IS_MULTI_CONDITIONS_INSERT        "is_multi_conditions_insert"
#define N_IS_MULTI_INSERT_FIRST             "is_multi_insert_first"
#define N_MULTI_VALUES_DESC                 "multi_values_desc"
#define N_MULTI_VALUE_VECTORS               "multi_value_vectors"
#define N_MULTI_INSERT_COL_CONV_FUNCS       "multi_insert_col_conv_funcs"
#define N_UNISTR                            "unistr"
#define N_ASCIISTR                          "asciistr"
#define N_ROWID_TO_CHAR                     "rowidtochar"
#define N_ROWID_TO_NCHAR                    "rowidtonchar"
#define N_CHAR_TO_ROWID                     "chartorowid"
#define N_OUTPUT_PACK                       "output_pack"
#define N_BENCHMARK                         "benchmark"
#define N_WEIGHT_STRING                     "weight_string"
#define N_DML_EVENT                         "dml_event"
#define N_TO_BASE64                         "to_base64"
#define N_FROM_BASE64                       "from_base64"
#define N_NLSSORT                           "nlssort"
#define N_JSON_OBJECT                       "json_object"
#define N_JSON_EXTRACT                      "json_extract"
#define N_JSON_CONTAINS                     "json_contains"
#define N_JSON_CONTAINS_PATH                "json_contains_path"
#define N_JSON_DEPTH                        "json_depth"
#define N_JSON_KEYS                         "json_keys"
#define N_JSON_ARRAY                        "json_array"
#define N_JSON_QUOTE                        "json_quote"
#define N_JSON_UNQUOTE                      "json_unquote"
#define N_JSON_OVERLAPS                     "json_overlaps"
#define N_JSON_VALID                        "json_valid"
#define N_JSON_REMOVE                       "json_remove"
#define N_JSON_SEARCH                       "json_search"
#define N_JSON_ARRAY_APPEND                 "json_array_append"
#define N_JSON_ARRAY_INSERT                 "json_array_insert"
#define N_JSON_VALUE                        "json_value"
#define N_JSON_REPLACE                      "json_replace"
#define N_JSON_TYPE                         "json_type"
#define N_JSON_LENGTH                       "json_length"
#define N_JSON_INSERT                       "json_insert"
#define N_JSON_STORAGE_SIZE                 "json_storage_size"
#define N_JSON_STORAGE_FREE                 "json_storage_free"
#define N_JSON_SET                          "json_set"
#define N_JSON_MERGE_PRESERVE               "json_merge_preserve"
#define N_JSON_MERGE                        "json_merge"
#define N_JSON_MERGE_PATCH                  "json_merge_patch"
#define N_JSON_PRETTY                       "json_pretty"
#define N_JSON_MEMBER_OF                    "json_member_of"
#define N_IS_JSON                           "is_json"
#define N_JSON_EQUAL                        "json_equal"
#define N_JSON_QUERY                        "json_query"
#define N_JSON_EXISTS                       "json_exists"

#define N_POINT                             "point"
#define N_LINESTRING                        "linestring"
#define N_MULTIPOINT                        "multipoint"
#define N_MULTILINESTRING                   "multilinestring"
#define N_POLYGON                           "polygon"
#define N_MULTIPOLYGON                      "multipolygon"
#define N_GEOMCOLLECTION                    "geomcollection"
#define N_GEOMETRYCOLLECTION                "geometrycollection"
#define N_ST_GEOMFROMTEXT                   "st_geomfromtext"
#define N_ST_GEOMETRYFROMTEXT               "st_geometryfromtext"
#define N_ST_INTERSECTION                   "st_intersection"
#define N_ST_AREA                           "st_area"
#define N_ST_INTERSECTS                     "st_intersects"
#define N_ST_X                              "st_x"
#define N_ST_Y                              "st_y"
#define N_ST_LATITUDE                       "st_latitude"
#define N_ST_LONGITUDE                      "st_longitude"
#define N_ST_TRANSFORM                      "st_transform"
#define N_PRIV_ST_COVERS                    "_st_covers"
#define N_PRIV_ST_TRANSFORM                 "_st_transform"
#define N_PRIV_ST_BESTSRID                  "_st_bestsrid"
#define N_ST_ASTEXT                         "st_astext"
#define N_ST_ASWKT                          "st_aswkt"
#define N_ST_BUFFER_STRATEGY                "st_buffer_strategy"
#define N_ST_BUFFER                         "st_buffer"
#define N_PRIV_ST_BUFFER                    "_st_buffer"
#define N_SPATIAL_CELLID                    "spatial_cellid"
#define N_SPATIAL_MBR                       "spatial_mbr"
#define N_PRIV_ST_GEOMFROMEWKB              "_st_geomfromewkb"
#define N_ST_GEOMFROMWKB                    "st_geomfromwkb"
#define N_ST_GEOMETRYFROMWKB                "st_geometryfromwkb"
#define N_PRIV_ST_GEOMFROMEWKT              "_st_geomfromewkt"
#define N_PRIV_ST_ASEWKT                    "_st_asewkt"
#define N_ST_DISTANCE                       "st_distance"
#define N_ST_SRID                           "st_srid"
#define N_PRIV_ST_SETSRID                   "_st_setsrid"
#define N_PRIV_ST_POINT                     "_st_point"
#define N_PRIV_ST_GEOGFROMTEXT              "_st_geogfromtext"
#define N_PRIV_ST_GEOGRAPHYFROMTEXT         "_st_geographyfromtext"
#define N_ST_ISVALID                        "st_isvalid"
#define N_PRIV_ST_DWITHIN                   "_st_dwithin"
#define N_ST_ASWKB                          "st_aswkb"
#define N_PRIV_ST_ASEWKB                    "_st_asewkb"
#define N_ST_ASBINARY                       "st_asbinary"
#define N_ST_DISTANCE_SPHERE                "st_distance_sphere"
#define N_ST_CONTAINS                       "st_contains"
#define N_ST_WITHIN                         "st_within"
#define N_SQL_MODE_CONVERT                  "sql_mode_convert"
#define N_NLS_INITCAP                       "nls_initcap"
#endif //OCEANBASE_LIB_OB_NAME_DEF_H_
