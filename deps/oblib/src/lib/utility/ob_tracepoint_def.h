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

#ifdef GLOBAL_ERRSIM_POINT_DEF
GLOBAL_ERRSIM_POINT_DEF(1, EN_1, "");
GLOBAL_ERRSIM_POINT_DEF(2, EN_2, "");
GLOBAL_ERRSIM_POINT_DEF(3, EN_3, "");
GLOBAL_ERRSIM_POINT_DEF(4, EN_4, "");
GLOBAL_ERRSIM_POINT_DEF(5, EN_5, "");
GLOBAL_ERRSIM_POINT_DEF(6, EN_6, "");
GLOBAL_ERRSIM_POINT_DEF(7, EN_7, "");
GLOBAL_ERRSIM_POINT_DEF(8, EN_8, "Used to simulate the scenario of failure to write temporary files");
GLOBAL_ERRSIM_POINT_DEF(9, EN_9, "");
GLOBAL_ERRSIM_POINT_DEF(10, EN_IS_LOG_SYNC, "");
GLOBAL_ERRSIM_POINT_DEF(11, EN_POST_ADD_REPILICA_MC, "");
GLOBAL_ERRSIM_POINT_DEF(12, EN_MIGRATE_FETCH_MACRO_BLOCK, "");
GLOBAL_ERRSIM_POINT_DEF(13, EN_WRITE_BLOCK, "");
GLOBAL_ERRSIM_POINT_DEF(14, EN_COMMIT_SLOG, "");
GLOBAL_ERRSIM_POINT_DEF(15, EN_SCHEDULE_INDEX_DAG, "");
GLOBAL_ERRSIM_POINT_DEF(16, EN_INDEX_LOCAL_SORT_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(17, EN_INDEX_MERGE_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(18, EN_INDEX_WRITE_BLOCK, "");
GLOBAL_ERRSIM_POINT_DEF(19, EN_INDEX_COMMIT_SLOG, "");
GLOBAL_ERRSIM_POINT_DEF(20, EN_CHECK_CAN_DO_MERGE, "");
GLOBAL_ERRSIM_POINT_DEF(21, EN_SCHEDULE_MERGE, "");
GLOBAL_ERRSIM_POINT_DEF(22, EN_MERGE_MACROBLOCK, "");
GLOBAL_ERRSIM_POINT_DEF(23, EN_MERGE_CHECKSUM, "");
GLOBAL_ERRSIM_POINT_DEF(24, EN_MERGE_FINISH, "");
GLOBAL_ERRSIM_POINT_DEF(25, EN_IO_SETUP, "");
GLOBAL_ERRSIM_POINT_DEF(26, EN_FORCE_WRITE_SSTABLE_SECOND_INDEX, "");
GLOBAL_ERRSIM_POINT_DEF(27, EN_SCHEDULE_MIGRATE, "");
GLOBAL_ERRSIM_POINT_DEF(28, EN_TRANS_AFTER_COMMIT, "");
GLOBAL_ERRSIM_POINT_DEF(29, EN_CHANGE_SCHEMA_VERSION_TO_ZERO, "");
GLOBAL_ERRSIM_POINT_DEF(30, EN_POST_REMOVE_REPLICA_MC_MSG, "");
GLOBAL_ERRSIM_POINT_DEF(31, EN_POST_ADD_REPLICA_MC_MSG, "");
GLOBAL_ERRSIM_POINT_DEF(32, EN_CHECK_SUB_MIGRATION_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(33, EN_POST_GET_MEMBER_LIST_MSG, "");
GLOBAL_ERRSIM_POINT_DEF(34, EN_WRITE_CHECKPOIRNT, "");
GLOBAL_ERRSIM_POINT_DEF(35, EN_MERGE_SORT_READ_MSG, "");
GLOBAL_ERRSIM_POINT_DEF(36, EN_IO_SUBMIT, "");
GLOBAL_ERRSIM_POINT_DEF(37, EN_IO_GETEVENTS, "");
GLOBAL_ERRSIM_POINT_DEF(38, EN_TRANS_LEADER_ACTIVE, "");
GLOBAL_ERRSIM_POINT_DEF(39, EN_UNIT_MANAGER, "");
GLOBAL_ERRSIM_POINT_DEF(40, EN_IO_CANCEL, "");
GLOBAL_ERRSIM_POINT_DEF(41, EN_REPLAY_ROW, "");
GLOBAL_ERRSIM_POINT_DEF(42, EN_BIG_ROW_REPLAY_FOR_MINORING, "");
GLOBAL_ERRSIM_POINT_DEF(43, EN_START_STMT_INTERFACE_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(44, EN_START_PARTICIPANT_INTERFACE_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(45, EN_END_PARTICIPANT_INTERFACE_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(46, EN_END_STMT_INTERFACE_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(47, EN_GET_GTS_LEADER, "");
GLOBAL_ERRSIM_POINT_DEF(48, ALLOC_LOG_ID_AND_TIMESTAMP_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(49, AFTER_MIGRATE_FINISH_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(52, EN_VALID_MIGRATE_SRC, "");
GLOBAL_ERRSIM_POINT_DEF(53, EN_BALANCE_TASK_EXE_ERR, "");
GLOBAL_ERRSIM_POINT_DEF(54, EN_ADD_REBUILD_PARENT_SRC, "");
GLOBAL_ERRSIM_POINT_DEF(55, EN_BAD_BLOCK_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(56, EN_ADD_RESTORE_TASK_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(57, EN_CTAS_FAIL_NO_DROP_ERROR, "Used to simulate the scenario that create table as select failed and then drop internal table failed");
GLOBAL_ERRSIM_POINT_DEF(58, EN_IO_CHANNEL_QUEUE_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(59, EN_GET_SCHE_CTX_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(60, EN_CLOG_RESTORE_REPLAYED_LOG, "");
GLOBAL_ERRSIM_POINT_DEF(61, EN_GEN_REBUILD_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(62, EN_IO_HANG_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(63, EN_CREATE_TENANT_TRANS_ONE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(64, EN_CREATE_TENANT_TRANS_TWO_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(65, EN_DELAY_REPLAY_SOURCE_SPLIT_LOG, "");
GLOBAL_ERRSIM_POINT_DEF(66, EN_BLOCK_SPLIT_PROGRESS_RESPONSE, "");
GLOBAL_ERRSIM_POINT_DEF(67, EN_RPC_ENCODE_SEGMENT_DATA_ERR, "");
GLOBAL_ERRSIM_POINT_DEF(68, EN_RPC_ENCODE_RAW_DATA_ERR, "");
GLOBAL_ERRSIM_POINT_DEF(69, EN_RPC_DECODE_COMPRESS_DATA_ERR, "");
GLOBAL_ERRSIM_POINT_DEF(70, EN_RPC_DECODE_RAW_DATA_ERR, "");
GLOBAL_ERRSIM_POINT_DEF(71, EN_BLOCK_SHUTDOWN_PARTITION, "");
GLOBAL_ERRSIM_POINT_DEF(72, EN_BLOCK_SPLIT_SOURCE_PARTITION, "");
GLOBAL_ERRSIM_POINT_DEF(73, EN_BLOCK_SUBMIT_SPLIT_SOURCE_LOG, "");
GLOBAL_ERRSIM_POINT_DEF(74, EN_BLOCK_SPLIT_DEST_PARTITION, "");
GLOBAL_ERRSIM_POINT_DEF(75, EN_CREATE_TENANT_TRANS_THREE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(76, EN_ALTER_CLUSTER_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(77, EN_STANDBY_REPLAY_SCHEMA_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(78, EN_STANDBY_REPLAY_CREATE_TABLE_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(79, EN_STANDBY_REPLAY_CREATE_TENANT_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(80, EN_STANDBY_REPLAY_CREATE_USER_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(81, EN_CREATE_TENANT_BEFORE_PERSIST_MEMBER_LIST, "");
GLOBAL_ERRSIM_POINT_DEF(82, EN_CREATE_TENANT_END_PERSIST_MEMBER_LIST, "");
GLOBAL_ERRSIM_POINT_DEF(83, EN_BROADCAST_CLUSTER_STATUS_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(84, EN_SET_FREEZE_INFO_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(85, EN_UPDATE_MAJOR_SCHEMA_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(86, EN_RENEW_SNAPSHOT_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(87, EN_FOLLOWER_UPDATE_FREEZE_INFO_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(88, EN_PARTITION_ITERATOR_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(89, EN_REFRESH_INCREMENT_SCHEMA_PHASE_THREE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(90, EN_MIGRATE_LOGIC_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(91, EN_REPLAY_ADD_PARTITION_TO_PG_CLOG, "");
GLOBAL_ERRSIM_POINT_DEF(92, EN_REPLAY_ADD_PARTITION_TO_PG_CLOG_AFTER_CREATE_SSTABLE, "");
GLOBAL_ERRSIM_POINT_DEF(93, EN_BEFORE_RENEW_SNAPSHOT_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(94, EN_BUILD_INDEX_RELEASE_SNAPSHOT_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(95, EN_CREATE_PG_PARTITION_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(96, EN_PUSH_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(97, EN_PUSH_REFERENCE_TABLE_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(98, EN_SLOG_WAIT_FLUSH_LOG, "");
GLOBAL_ERRSIM_POINT_DEF(99, EN_SET_MEMBER_LIST_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(100, EN_CREATE_TABLE_TRANS_END_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(101, EN_ABROT_INDEX_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(102, EN_DELAY_REPLAY_SOURCE_SPLIT_LOG_R_REPLICA, "");
GLOBAL_ERRSIM_POINT_DEF(103, EN_SKIP_GLOBAL_SSTABLE_SCHEMA_VERSION, "");
GLOBAL_ERRSIM_POINT_DEF(104, EN_STOP_ROOT_INSPECTION, "");
GLOBAL_ERRSIM_POINT_DEF(105, EN_DROP_TENANT_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(106, EN_SKIP_DROP_MEMTABLE, "");
GLOBAL_ERRSIM_POINT_DEF(107, EN_SKIP_DROP_PG_PARTITION, "");
GLOBAL_ERRSIM_POINT_DEF(109, EN_OBSERVER_CREATE_PARTITION_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(110, EN_CREATE_PARTITION_WITH_OLD_MAJOR_TS, "");
GLOBAL_ERRSIM_POINT_DEF(111, EN_PREPARE_SPLIT_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(112, EN_REPLAY_SOURCE_SPLIT_LOG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(113, EN_SAVE_SPLIT_STATE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(114, EN_FORCE_REFRESH_TABLE, "");
GLOBAL_ERRSIM_POINT_DEF(116, EN_REPLAY_SPLIT_DEST_LOG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(117, EN_PROCESS_TO_PRIMARY_ERR, "");
GLOBAL_ERRSIM_POINT_DEF(118, EN_CREATE_PG_AFTER_CREATE_SSTBALES, "");
GLOBAL_ERRSIM_POINT_DEF(119, EN_CREATE_PG_AFTER_REGISTER_TRANS_SERVICE, "");
GLOBAL_ERRSIM_POINT_DEF(120, EN_CREATE_PG_AFTER_REGISTER_ELECTION_MGR, "");
GLOBAL_ERRSIM_POINT_DEF(121, EN_CREATE_PG_AFTER_ADD_PARTITIONS_TO_MGR, "");
GLOBAL_ERRSIM_POINT_DEF(122, EN_CREATE_PG_AFTER_ADD_PARTITIONS_TO_REPLAY_ENGINE, "");
GLOBAL_ERRSIM_POINT_DEF(123, EN_CREATE_PG_AFTER_BATCH_START_PARTITION_ELECTION, "");
GLOBAL_ERRSIM_POINT_DEF(124, EN_BACKUP_MACRO_BLOCK_SUBTASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(125, EN_BACKUP_REPORT_RESULT_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(126, EN_RESTORE_UPDATE_PARTITION_META_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(127, EN_BACKUP_FILTER_TABLE_BY_SCHEMA, "");
GLOBAL_ERRSIM_POINT_DEF(128, EN_FORCE_DFC_BLOCK, "");
GLOBAL_ERRSIM_POINT_DEF(129, EN_SERVER_PG_META_WRITE_HALF_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(130, EN_SERVER_TENANT_FILE_SUPER_BLOCK_WRITE_HALF_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(131, EN_DTL_ONE_ROW_ONE_BUFFER, "");
GLOBAL_ERRSIM_POINT_DEF(132, EN_LOG_ARCHIVE_PUSH_LOG_FAILED, "");

GLOBAL_ERRSIM_POINT_DEF(133, EN_BACKUP_DATA_VERSION_GAP_OVER_LIMIT, "");
GLOBAL_ERRSIM_POINT_DEF(134, EN_LOG_ARHIVE_SCHEDULER_INTERRUPT, "");
GLOBAL_ERRSIM_POINT_DEF(135, EN_BACKUP_IO_LIST_FILE, "");
GLOBAL_ERRSIM_POINT_DEF(136, EN_BACKUP_IO_IS_EXIST, "");
GLOBAL_ERRSIM_POINT_DEF(137, EN_BACKUP_IO_GET_FILE_LENGTH, "");
GLOBAL_ERRSIM_POINT_DEF(138, EN_BACKUP_IO_BEFORE_DEL_FILE, "");
GLOBAL_ERRSIM_POINT_DEF(139, EN_BACKUP_IO_AFTER_DEL_FILE, "");
GLOBAL_ERRSIM_POINT_DEF(140, EN_BACKUP_IO_BEFORE_MKDIR, "");
GLOBAL_ERRSIM_POINT_DEF(141, EN_BACKUP_IO_AFTER_MKDIR, "");
GLOBAL_ERRSIM_POINT_DEF(142, EN_BACKUP_IO_UPDATE_FILE_MODIFY_TIME, "");
GLOBAL_ERRSIM_POINT_DEF(143, EN_BACKUP_IO_BEFORE_WRITE_SINGLE_FILE, "");
GLOBAL_ERRSIM_POINT_DEF(144, EN_BACKUP_IO_AFTER_WRITE_SINGLE_FILE, "");
GLOBAL_ERRSIM_POINT_DEF(145, EN_BACKUP_IO_READER_OPEN, "");
GLOBAL_ERRSIM_POINT_DEF(146, EN_BACKUP_IO_READER_PREAD, "");
GLOBAL_ERRSIM_POINT_DEF(147, EN_BACKUP_IO_WRITE_OPEN, "");
GLOBAL_ERRSIM_POINT_DEF(148, EN_BACKUP_IO_WRITE_WRITE, "");
GLOBAL_ERRSIM_POINT_DEF(149, EN_BACKUP_IO_APPENDER_OPEN, "");
GLOBAL_ERRSIM_POINT_DEF(150, EN_BACKUP_IO_APPENDER_WRITE, "");
GLOBAL_ERRSIM_POINT_DEF(151, EN_ROOT_BACKUP_MAX_GENERATE_NUM, "");
GLOBAL_ERRSIM_POINT_DEF(152, EN_ROOT_BACKUP_NEED_SWITCH_TENANT, "");
GLOBAL_ERRSIM_POINT_DEF(153, EN_BACKUP_FILE_APPENDER_CLOSE, "");
GLOBAL_ERRSIM_POINT_DEF(154, EN_RESTORE_MACRO_CRC_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(155, EN_BACKUP_DELETE_HANDLE_LS_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(156, EN_BACKUP_DELETE_MARK_DELETING, "");
GLOBAL_ERRSIM_POINT_DEF(157, EN_RESTORE_FETCH_CLOG_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(158, EN_BACKUP_LEASE_CAN_TAKEOVER, "");
GLOBAL_ERRSIM_POINT_DEF(159, EN_BACKUP_EXTERN_INFO_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(160, EN_INCREMENTAL_BACKUP_NUM, "");
GLOBAL_ERRSIM_POINT_DEF(161, EN_LOG_ARCHIVE_BEFORE_PUSH_LOG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(162, EN_BACKUP_META_INDEX_BUFFER_NOT_COMPLETED, "");
GLOBAL_ERRSIM_POINT_DEF(163, EN_BACKUP_MACRO_INDEX_BUFFER_NOT_COMPLETED, "");
GLOBAL_ERRSIM_POINT_DEF(164, EN_LOG_ARCHIVE_DATA_BUFFER_NOT_COMPLETED, "");
GLOBAL_ERRSIM_POINT_DEF(165, EN_LOG_ARCHIVE_INDEX_BUFFER_NOT_COMPLETED, "");
GLOBAL_ERRSIM_POINT_DEF(166, EN_FILE_SYSTEM_RENAME_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(167, EN_BACKUP_OBSOLETE_INTERVAL, "");
GLOBAL_ERRSIM_POINT_DEF(168, EN_BACKUP_BACKUP_LOG_ARCHIVE_INTERRUPTED, "");
GLOBAL_ERRSIM_POINT_DEF(169, EN_BACKUP_BACKUPSET_EXTERN_INFO_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(170, EN_BACKUP_SCHEDULER_GET_SCHEMA_VERSION_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(171, EN_BACKUP_BACKUPSET_FILE_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(172, EN_LOG_ARCHIVE_RESTORE_ACCUM_CHECKSUM_TAMPERED, "");
GLOBAL_ERRSIM_POINT_DEF(173, EN_BACKUP_BACKUPPIECE_FILE_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(174, EN_BACKUP_RS_BLOCK_FROZEN_PIECE, "");
GLOBAL_ERRSIM_POINT_DEF(175, EN_LOG_ARCHIVE_BLOCK_SWITCH_PIECE, "");
GLOBAL_ERRSIM_POINT_DEF(176, EN_BACKUP_ARCHIVELOG_RPC_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(177, EN_BACKUP_AFTER_UPDATE_EXTERNAL_ROUND_INFO_FOR_USER, "");
GLOBAL_ERRSIM_POINT_DEF(178, EN_BACKUP_AFTER_UPDATE_EXTERNAL_ROUND_INFO_FOR_SYS, "");
GLOBAL_ERRSIM_POINT_DEF(179, EN_BACKUP_AFTER_UPDATE_EXTERNAL_BOTH_PIECE_INFO_FOR_USER, "");
GLOBAL_ERRSIM_POINT_DEF(180, EN_BACKUP_AFTER_UPDATE_EXTERNAL_BOTH_PIECE_INFO_FOR_SYS, "");
GLOBAL_ERRSIM_POINT_DEF(181, EN_BACKUP_BACKUPPIECE_FINISH_UPDATE_EXTERN_AND_INNER_INFO, "");
GLOBAL_ERRSIM_POINT_DEF(182, EN_BACKUP_BACKUPPIECE_DO_SCHEDULE, "");
GLOBAL_ERRSIM_POINT_DEF(183, EN_STOP_TENANT_LOG_ARCHIVE_BACKUP, "");
GLOBAL_ERRSIM_POINT_DEF(184, EN_BACKUP_SERVER_DISK_IS_FULL, "");
GLOBAL_ERRSIM_POINT_DEF(185, EN_CHANGE_TENANT_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(186, EN_BACKUP_PERSIST_LS_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(189, EN_BACKUP_VALIDATE_DO_FINISH, "");
GLOBAL_ERRSIM_POINT_DEF(190, EN_BACKUP_SYS_META_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(191, EN_BACKUP_SYS_TABLET_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(192, EN_BACKUP_DATA_TABLET_MINOR_SSTABLE_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(193, EN_BACKUP_DATA_TABLET_MAJOR_SSTABLE_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(194, EN_BACKUP_BUILD_LS_LEVEL_INDEX_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(195, EN_BACKUP_BUILD_TENANT_LEVEL_INDEX_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(196, EN_BACKUP_PREFETCH_BACKUP_INFO_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(197, EN_BACKUP_COMPLEMENT_LOG_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(198, EN_BACKUP_USER_META_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(199, EN_BACKUP_PREPARE_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(200, EN_BACKUP_CHECK_TABLET_CONTINUITY_FAILED, "");
// 下面请从201开始
GLOBAL_ERRSIM_POINT_DEF(201, EN_CHECK_STANDBY_CLUSTER_SCHEMA_CONDITION, "");
GLOBAL_ERRSIM_POINT_DEF(202, EN_ALLOCATE_LOB_BUF_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(203, EN_ALLOCATE_DESERIALIZE_LOB_BUF_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(204, EN_ENCRYPT_ALLOCATE_HASHMAP_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(205, EN_ENCRYPT_ALLOCATE_ROW_BUF_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(206, EN_ENCRYPT_GET_MASTER_KEY_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(207, EN_DECRYPT_ALLOCATE_ROW_BUF_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(208, EN_DECRYPT_GET_MASTER_KEY_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(209, EN_FAST_MIGRATE_CHANGE_MEMBER_LIST_NOT_BEGIN, "");
GLOBAL_ERRSIM_POINT_DEF(210, EN_FAST_MIGRATE_CHANGE_MEMBER_LIST_AFTER_REMOVE, "");
GLOBAL_ERRSIM_POINT_DEF(211, EN_FAST_MIGRATE_CHANGE_MEMBER_LIST_SUCCESS_BUT_TIMEOUT, "");
GLOBAL_ERRSIM_POINT_DEF(212, EN_SCHEDULE_DATA_MINOR_MERGE, "");
GLOBAL_ERRSIM_POINT_DEF(213, EN_LOG_SYNC_SLOW, "");
GLOBAL_ERRSIM_POINT_DEF(214, EN_WRITE_CONFIG_FILE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(215, EN_INVALID_ADDR_WEAK_READ_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(216, EN_STACK_OVERFLOW_CHECK_EXPR_STACK_SIZE, "Discarded");
GLOBAL_ERRSIM_POINT_DEF(217, EN_ENABLE_PDML_ALL_FEATURE, "");
// slog checkpoint错误模拟占坑 218-230
GLOBAL_ERRSIM_POINT_DEF(218, EN_SLOG_CKPT_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(219, EN_FAST_RECOVERY_AFTER_ALLOC_FILE, "");
GLOBAL_ERRSIM_POINT_DEF(220, EN_FAST_MIGRATE_ADD_MEMBER_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(221, EN_FAST_RECOVERY_BEFORE_ADD_MEMBER, "");
GLOBAL_ERRSIM_POINT_DEF(222, EN_FAST_RECOVERY_AFTER_ADD_MEMBER, "");
GLOBAL_ERRSIM_POINT_DEF(223, EN_FAST_RECOVERY_AFTER_REMOVE_MEMBER, "");
GLOBAL_ERRSIM_POINT_DEF(224, EN_OFS_IO_SUBMIT, "For debugging purposes, deprecated.");
GLOBAL_ERRSIM_POINT_DEF(225, EN_MIGRATE_ADD_PARTITION_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(231, EN_PRINT_QUERY_SQL, "");
GLOBAL_ERRSIM_POINT_DEF(232, EN_ADD_NEW_PG_TO_PARTITION_SERVICE, "");
GLOBAL_ERRSIM_POINT_DEF(233, EN_DML_DISABLE_RANDOM_RESHUFFLE, "");
GLOBAL_ERRSIM_POINT_DEF(234, EN_RESIZE_PHYSICAL_FILE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(235, EN_ALLOCATE_RESIZE_MEMORY_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(236, EN_WRITE_SUPER_BLOCK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(237, EN_GC_FAILED_PARTICIPANTS, "");
GLOBAL_ERRSIM_POINT_DEF(238, EN_SSL_INVITE_NODES_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(239, EN_ADD_TRIGGER_SKIP_MAP, "");
GLOBAL_ERRSIM_POINT_DEF(240, EN_DEL_TRIGGER_SKIP_MAP, "");
GLOBAL_ERRSIM_POINT_DEF(241, EN_RESET_FREE_MEMORY, "");
GLOBAL_ERRSIM_POINT_DEF(242, EN_BKGD_TASK_REPORT_COMPLETE, "");
GLOBAL_ERRSIM_POINT_DEF(243, EN_BKGD_TRANSMIT_CHECK_STATUS_PER_ROW, "");
GLOBAL_ERRSIM_POINT_DEF(244, EN_OPEN_REMOTE_ASYNC_EXECUTION, "");
GLOBAL_ERRSIM_POINT_DEF(245, EN_BACKUP_DELETE_EXCEPTION_HANDLING, "");
GLOBAL_ERRSIM_POINT_DEF(246, EN_SORT_IMPL_FORCE_DO_DUMP, "Used to simulate the scenario of failure to write temporary files");
GLOBAL_ERRSIM_POINT_DEF(247, EN_ENFORCE_PUSH_DOWN_WF, "Used to enforce pushdown window function regardless of ndv and dop");
GLOBAL_ERRSIM_POINT_DEF(248, EN_SORT_IMPL_TOPN_EAGER_FILTER, "Used to control whether to use eager filtering to accelerate the top-N operator");
//
GLOBAL_ERRSIM_POINT_DEF(250, EN_TRANS_SHARED_LOCK_CONFLICT, "");
GLOBAL_ERRSIM_POINT_DEF(251, EN_HASH_JOIN_OPTION, "Cache aware hash join switch (value & 0x2), and  extra bloom filter in hash join switch (value & 0x4).");
GLOBAL_ERRSIM_POINT_DEF(252, EN_SET_DISABLE_HASH_JOIN_BATCH, "");
GLOBAL_ERRSIM_POINT_DEF(253, EN_INNER_SQL_CONN_LEAK_CHECK, "");
GLOBAL_ERRSIM_POINT_DEF(254, EN_ADAPTIVE_GROUP_BY_SMALL_CACHE, "Used to simulate a scenario where hash gby quickly enters the adaptive state");

// only work for remote execute
GLOBAL_ERRSIM_POINT_DEF(255, EN_DISABLE_REMOTE_EXEC_WITH_PLAN, "");
GLOBAL_ERRSIM_POINT_DEF(256, EN_REMOTE_EXEC_ERR, "");

GLOBAL_ERRSIM_POINT_DEF(260, EN_XA_PREPARE_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(261, EN_XA_UPDATE_COORD_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(262, EN_XA_PREPARE_RESP_LOST, "");
GLOBAL_ERRSIM_POINT_DEF(263, EN_XA_RPC_TIMEOUT, "");
GLOBAL_ERRSIM_POINT_DEF(264, EN_XA_COMMIT_ABORT_RESP_LOST, "");
GLOBAL_ERRSIM_POINT_DEF(265, EN_XA_1PC_RESP_LOST, "");
GLOBAL_ERRSIM_POINT_DEF(266, EN_DISK_ERROR, "Deprecated variable");


GLOBAL_ERRSIM_POINT_DEF(267, EN_CLOG_DUMP_ILOG_MEMSTORE_RENAME_FAILURE, "");
GLOBAL_ERRSIM_POINT_DEF(268, EN_CLOG_ILOG_MEMSTORE_ALLOC_MEMORY_FAILURE, "");
GLOBAL_ERRSIM_POINT_DEF(269, EN_CLOG_LOG_NOT_IN_SW, "");
GLOBAL_ERRSIM_POINT_DEF(270, EN_CLOG_PARTITION_IS_NOT_SYNC, "");
GLOBAL_ERRSIM_POINT_DEF(271, EN_CLOG_LOG_NOT_IN_ILOG_STORAGE, "");
GLOBAL_ERRSIM_POINT_DEF(272, EN_CLOG_SW_OUT_OF_RANGE, "");
GLOBAL_ERRSIM_POINT_DEF(273, EN_DFC_FACTOR, "");
GLOBAL_ERRSIM_POINT_DEF(274, EN_LOGSERVICE_IO_TIMEOUT, "");

GLOBAL_ERRSIM_POINT_DEF(275, EN_PARTICIPANTS_SIZE_OVERFLOW, "");
GLOBAL_ERRSIM_POINT_DEF(276, EN_UNDO_ACTIONS_SIZE_OVERFLOW, "");
GLOBAL_ERRSIM_POINT_DEF(277, EN_PART_PLUS_UNDO_OVERFLOW, "");
GLOBAL_ERRSIM_POINT_DEF(278, EN_HANDLE_PREPARE_MESSAGE_EAGAIN, "");
GLOBAL_ERRSIM_POINT_DEF(279, EN_RC_ONLY_LEADER_TO_LEADER, "");
GLOBAL_ERRSIM_POINT_DEF(280, EN_REPLAY_SERVICE_SUBMIT_TASK_SLEEP, "");

//simulate DAS errors 301-350
GLOBAL_ERRSIM_POINT_DEF(301, EN_DAS_SCAN_RESULT_OVERFLOW, "");
GLOBAL_ERRSIM_POINT_DEF(302, EN_DAS_DML_BUFFER_OVERFLOW, "");
GLOBAL_ERRSIM_POINT_DEF(303, EN_DAS_SIMULATE_OPEN_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(304, EN_DAS_WRITE_ROW_LIST_LEN, "");
GLOBAL_ERRSIM_POINT_DEF(305, EN_DAS_SIMULATE_VT_CREATE_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(306, EN_DAS_SIMULATE_LOOKUPOP_INIT_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(307, EN_DAS_SIMULATE_ASYNC_RPC_TIMEOUT, "");
GLOBAL_ERRSIM_POINT_DEF(308, EN_DAS_SIMULATE_DUMP_WRITE_BUFFER, "");
GLOBAL_ERRSIM_POINT_DEF(309, EN_DAS_SIMULATE_AGG_TASK_BUFF_LIMIT, "");
GLOBAL_ERRSIM_POINT_DEF(310, EN_DAS_ALL_PARALLEL_TASK_MEM_LIMIT, "");
GLOBAL_ERRSIM_POINT_DEF(311, EN_DAS_SIMULATE_GROUP_SIZE, "");
GLOBAL_ERRSIM_POINT_DEF(312, EN_DAS_SIMULATE_DAS_TASK_SIZE, "");
GLOBAL_ERRSIM_POINT_DEF(313, EN_DAS_SIMULATE_AGG_TASK_MEM_LIMIT, "");
GLOBAL_ERRSIM_POINT_DEF(314, EN_DAS_SIMULATE_AGG_TASK_RETRY_CODE, "");
GLOBAL_ERRSIM_POINT_DEF(315, EN_DAS_GROUP_RESCAN_TEST_MODE, "");
GLOBAL_ERRSIM_POINT_DEF(316, EN_DAS_SIMULATE_MAX_ROWSETS, "");

GLOBAL_ERRSIM_POINT_DEF(351, EN_REPLAY_STORAGE_SCHEMA_FAILURE, "");
GLOBAL_ERRSIM_POINT_DEF(352, EN_SKIP_GET_STORAGE_SCHEMA, "");
GLOBAL_ERRSIM_POINT_DEF(353, EN_DISABLE_RICH_FORMAT_IN_STORAGE, "");

GLOBAL_ERRSIM_POINT_DEF(360, EN_PREVENT_SYNC_REPORT, "");
GLOBAL_ERRSIM_POINT_DEF(361, EN_PREVENT_ASYNC_REPORT, "");
GLOBAL_ERRSIM_POINT_DEF(362, EN_REBALANCE_TASK_RETRY, "");
GLOBAL_ERRSIM_POINT_DEF(363, EN_LOG_IDS_COUNT_ERROR, "");

GLOBAL_ERRSIM_POINT_DEF(364, EN_AMM_WASH_RATIO, "Calculate the maximum coarse granularity of washable size");
GLOBAL_ERRSIM_POINT_DEF(365, EN_ENABLE_THREE_STAGE_AGGREGATE, "");
GLOBAL_ERRSIM_POINT_DEF(366, EN_ROLLUP_ADAPTIVE_KEY_NUM, "");
GLOBAL_ERRSIM_POINT_DEF(367, EN_ENABLE_OP_OUTPUT_DATUM_CHECK, "Used to check whether the datum ptr of the operator output is valid");
GLOBAL_ERRSIM_POINT_DEF(368, EN_LEADER_STORAGE_ESTIMATION, "");

// SQL table_scan, index_look_up and other dml_op 400-500
GLOBAL_ERRSIM_POINT_DEF(400, EN_TABLE_LOOKUP_BATCH_ROW_COUNT, "");
GLOBAL_ERRSIM_POINT_DEF(401, EN_TABLE_REPLACE_BATCH_ROW_COUNT, "");
GLOBAL_ERRSIM_POINT_DEF(402, EN_TABLE_INSERT_UP_BATCH_ROW_COUNT, "");
GLOBAL_ERRSIM_POINT_DEF(403, EN_EXPLAIN_BATCHED_MULTI_STATEMENT, "");
GLOBAL_ERRSIM_POINT_DEF(404, EN_INS_MULTI_VALUES_BATCH_OPT, "");
GLOBAL_ERRSIM_POINT_DEF(405, EN_SQL_MEMORY_LABEL_HIGH64, "");
GLOBAL_ERRSIM_POINT_DEF(406, EN_SQL_MEMORY_LABEL_LOW64, "");
GLOBAL_ERRSIM_POINT_DEF(407, EN_SQL_MEMORY_DYNAMIC_LEAK_SIZE, "");

// DDL related 500-550
GLOBAL_ERRSIM_POINT_DEF(501, EN_DATA_CHECKSUM_DDL_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(502, EN_HIDDEN_CHECKSUM_DDL_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(503, EN_SUBMIT_INDEX_TASK_ERROR_BEFORE_STAT_RECORD, "");
GLOBAL_ERRSIM_POINT_DEF(504, EN_SUBMIT_INDEX_TASK_ERROR_AFTER_STAT_RECORD, "");
GLOBAL_ERRSIM_POINT_DEF(505, EN_BUILD_LOCAL_INDEX_WITH_CORRUPTED_DATA, "");
GLOBAL_ERRSIM_POINT_DEF(506, EN_BUILD_GLOBAL_INDEX_WITH_CORRUPTED_DATA, "");
GLOBAL_ERRSIM_POINT_DEF(509, EN_EARLY_RESPONSE_SCHEDULER, "");
GLOBAL_ERRSIM_POINT_DEF(510, EN_DDL_TASK_PROCESS_FAIL_STATUS, "");
GLOBAL_ERRSIM_POINT_DEF(511, EN_DDL_TASK_PROCESS_FAIL_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(512, EN_DDL_START_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(513, EN_DDL_COMPACT_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(514, EN_DDL_RELEASE_DDL_KV_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(515, EN_DDL_REPORT_CHECKSUM_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(516, EN_DDL_REPORT_REPLICA_BUILD_STATUS_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(517, EN_DDL_DIRECT_LOAD_WAIT_TABLE_LOCK_FAIL, "");
GLOBAL_ERRSIM_POINT_DEF(518, EN_DDL_LOBID_CACHE_SIZE_INJECTED, "");
GLOBAL_ERRSIM_POINT_DEF(519, EN_DDL_EXECUTE_FAILED, "");

// SQL Optimizer related 551-599
GLOBAL_ERRSIM_POINT_DEF(551, EN_EXPLAIN_GENERATE_PLAN_WITH_OUTLINE, "Used to enable outline validity check for explain query");
GLOBAL_ERRSIM_POINT_DEF(552, EN_ENABLE_AUTO_DOP_FORCE_PARALLEL_PLAN, "Used to generate parallel plan with random dop");
GLOBAL_ERRSIM_POINT_DEF(553, EN_GENERATE_PLAN_WITH_RECONSTRUCT_SQL, "wether to use reconstructed sql to generate plan");
GLOBAL_ERRSIM_POINT_DEF(554, EN_GENERATE_PLAN_WITH_NLJ, "");
GLOBAL_ERRSIM_POINT_DEF(555, EN_CHECK_OPERATOR_OUTPUT_ROWS, "");
GLOBAL_ERRSIM_POINT_DEF(556, EN_GENERATE_RANDOM_PLAN, "Whether the optimizer generates random plans");
GLOBAL_ERRSIM_POINT_DEF(557, EN_COALESCE_AGGR_IGNORE_COST, "");
GLOBAL_ERRSIM_POINT_DEF(558, EN_CHECK_REWRITE_ITER_CONVERGE, "Reporting error when rewrite iter nonconvergent");
GLOBAL_ERRSIM_POINT_DEF(559, EN_PRINT_CONSTRAINTS_INFO, "show constraints info when explain query plan");

// 600-700 For PX use
GLOBAL_ERRSIM_POINT_DEF(600, EN_PX_SQC_EXECUTE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(601, EN_PX_SQC_INIT_FAILED, "Used to simulate the scenario of failure to init sub query coordinator");
GLOBAL_ERRSIM_POINT_DEF(602, EN_PX_SQC_INIT_PROCESS_FAILED, "Inject error: let sqc init failed");
GLOBAL_ERRSIM_POINT_DEF(603, EN_PX_PRINT_TARGET_MONITOR_LOG, "whether print debug log of px target monitor module");
GLOBAL_ERRSIM_POINT_DEF(604, EN_PX_SQC_NOT_REPORT_TO_QC, "Inject error: let sqc not send finish message to qc");
GLOBAL_ERRSIM_POINT_DEF(605, EN_PX_QC_EARLY_TERMINATE, "Inject error: let PX coordinator quit fastly");
GLOBAL_ERRSIM_POINT_DEF(606, EN_PX_SINGLE_DFO_NOT_ERASE_DTL_INTERM_RESULT, "Inject error: skip the erase interm result process");
GLOBAL_ERRSIM_POINT_DEF(607, EN_PX_TEMP_TABLE_NOT_DESTROY_REMOTE_INTERM_RESULT, "Inject error: let interm result of temp table not be cleared");
GLOBAL_ERRSIM_POINT_DEF(608, EN_PX_NOT_ERASE_P2P_DH_MSG, "Inject error: let runtime filter msg not be erased by PX coordinator");
GLOBAL_ERRSIM_POINT_DEF(609, EN_PX_SLOW_PROCESS_SQC_FINISH_MSG, "Inject error: let PX slowly process the sqc finish message");
GLOBAL_ERRSIM_POINT_DEF(610, EN_PX_JOIN_FILTER_NOT_MERGE_MSG, "Inject error: let runtime filter skip the merge process.");
GLOBAL_ERRSIM_POINT_DEF(611, EN_PX_P2P_MSG_REG_DM_FAILED, "Inject error: let runtime filter failed to register into DM.");
GLOBAL_ERRSIM_POINT_DEF(612, EN_PX_JOIN_FILTER_HOLD_MSG, "Inject error: let runtime filter destroy later for a long time.");
GLOBAL_ERRSIM_POINT_DEF(613, EN_PX_DTL_TRACE_LOG_ENABLE, "Deprecated variable");
GLOBAL_ERRSIM_POINT_DEF(614, EN_PX_DISABLE_RUNTIME_FILTER_EXTRACT_QUERY_RANGE, "Switch: use to disable the feature runtime filter extracting query range");
GLOBAL_ERRSIM_POINT_DEF(615, EN_PX_MAX_IN_FILTER_QR_COUNT, "Switch: control the max number of query range extract by runtime in filter");
GLOBAL_ERRSIM_POINT_DEF(616, EN_PX_DISABLE_WHITE_RUNTIME_FILTER, "Switch: used to disable runtime filter pushdown as white filter.");
GLOBAL_ERRSIM_POINT_DEF(617, EN_PX_DISABLE_PD_TOPN_FILTER, "Switch: used to disable runtime topn filter pushdown.");
GLOBAL_ERRSIM_POINT_DEF(618, EN_PX_PD_TOPN_FILTER_IGNORE_TABLE_CARD, "Switch: allocate topn filter expr even if less table card");
GLOBAL_ERRSIM_POINT_DEF(619, EN_PX_SQC_HANDLER_INIT_FAILED, "Inject error: let sqc handler init failed");
// please add new trace point after 700 or before 600

// Compaction Related 700-750
GLOBAL_ERRSIM_POINT_DEF(700, EN_COMPACTION_DIAGNOSE_TABLE_STORE_UNSAFE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(701, EN_COMPACTION_DIAGNOSE_CANNOT_MAJOR, "");
GLOBAL_ERRSIM_POINT_DEF(702, EN_COMPACTION_MERGE_TASK, "");
GLOBAL_ERRSIM_POINT_DEF(703, EN_MEDIUM_COMPACTION_SUBMIT_CLOG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(704, EN_MEDIUM_COMPACTION_UPDATE_CUR_SNAPSHOT_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(705, EN_MEDIUM_REPLICA_CHECKSUM_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(706, EN_MEDIUM_CREATE_DAG, "");
GLOBAL_ERRSIM_POINT_DEF(707, EN_MEDIUM_VERIFY_GROUP_SKIP_SET_VERIFY, "");
GLOBAL_ERRSIM_POINT_DEF(708, EN_MEDIUM_VERIFY_GROUP_SKIP_COLUMN_CHECKSUM, "");
GLOBAL_ERRSIM_POINT_DEF(709, EN_SCHEDULE_MEDIUM_COMPACTION, "");
GLOBAL_ERRSIM_POINT_DEF(710, EN_SCHEDULE_MAJOR_GET_TABLE_SCHEMA, "");
GLOBAL_ERRSIM_POINT_DEF(711, EN_SKIP_INDEX_MAJOR, "");
GLOBAL_ERRSIM_POINT_DEF(712, EN_BUILD_DATA_MICRO_BLOCK, "");
GLOBAL_ERRSIM_POINT_DEF(713, EN_COMPACTION_CO_MERGE_EXE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(714, EN_COMPACTION_CO_MERGE_SCHEDULE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(715, EN_COMPACTION_MEDIUM_INIT_PARALLEL_RANGE, "");
GLOBAL_ERRSIM_POINT_DEF(716, EN_RS_USER_INDEX_CHECKSUM_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(717, EN_RS_CANT_GET_ALL_TABLET_CHECKSUM, "");
GLOBAL_ERRSIM_POINT_DEF(718, EN_SWAP_TABLET_IN_COMPACTION, "");
GLOBAL_ERRSIM_POINT_DEF(719, EN_COMPACTION_CO_MERGE_PREPARE_CTX_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(720, EN_COMPACTION_CO_MERGE_PREPARE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(721, EN_COMPACTION_CO_MERGE_PREPARE_MINOR_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(722, EN_COMPACTION_CO_MERGE_FINISH_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(723, EN_COMPACTION_ITER_TABLET_NOT_EXIST, "");
GLOBAL_ERRSIM_POINT_DEF(724, EN_COMPACTION_ITER_LS_NOT_EXIST, "");
GLOBAL_ERRSIM_POINT_DEF(725, EN_COMPACTION_ITER_INVALID_TABLET_ID, "");
GLOBAL_ERRSIM_POINT_DEF(726, EN_RS_CHECK_SPECIAL_TABLE, "");
GLOBAL_ERRSIM_POINT_DEF(727, EN_COMPACTION_REPORT_ADD_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(728, EN_COMPACTION_REPORT_PROCESS_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(729, EN_RS_CHECK_MERGE_PROGRESS, "");
GLOBAL_ERRSIM_POINT_DEF(730, EN_CAN_NOT_SCHEDULE_MINOR, "");
GLOBAL_ERRSIM_POINT_DEF(731, EN_SCHEDULE_MEDIUM_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(732, EN_SPECIAL_TABLE_HAVE_LARGER_SCN, "");
GLOBAL_ERRSIM_POINT_DEF(733, EN_COMPACTION_CO_PUSH_TABLES_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(734, EN_COMPACTION_CO_MERGE_PARTITION_LONG_TIME, "");
GLOBAL_ERRSIM_POINT_DEF(735, EN_COMPACTION_SCHEDULE_META_MERGE, "");
GLOBAL_ERRSIM_POINT_DEF(736, EN_COMPACTION_ESTIMATE_ROW_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(737, EN_COMPACTION_UPDATE_REPORT_SCN, "");
GLOBAL_ERRSIM_POINT_DEF(738, EN_CO_MREGE_DAG_READY_FOREVER, "");
GLOBAL_ERRSIM_POINT_DEF(739, EN_CO_MREGE_DAG_SCHEDULE_REST, "");
GLOBAL_ERRSIM_POINT_DEF(740, EN_COMPACTION_SCHEDULE_MEDIUM_MERGE_AFTER_MINI, "");
GLOBAL_ERRSIM_POINT_DEF(741, EN_COMPACTION_MEDIUM_INIT_LARGE_PARALLEL_RANGE, "");
GLOBAL_ERRSIM_POINT_DEF(742, EN_GET_TABLET_LS_PAIR_IN_RS, "");
GLOBAL_ERRSIM_POINT_DEF(743, EN_SHARED_STORAGE_COMPACTION_CHOOSE_EXEC_SVR, "");
GLOBAL_ERRSIM_POINT_DEF(744, EN_SHARED_STORAGE_SKIP_USER_TABLET_REFRESH, "");
GLOBAL_ERRSIM_POINT_DEF(745, EN_SHARED_STORAGE_SCHEULD_TABLET_IN_IDLE, "");
GLOBAL_ERRSIM_POINT_DEF(746, EN_SHARED_STORAGE_DONT_UPDATE_LS_STATE, "");
GLOBAL_ERRSIM_POINT_DEF(747, EN_MAKE_DATA_CKM_ERROR_BY_WRITE_WRONG_ROW, "change last datum of row into int(999) for making checksum error");
GLOBAL_ERRSIM_POINT_DEF(748, EN_COMPACTION_ITER_SET_BATCH_CNT, "");

// compaction end at 750

// please add new trace point after 750
GLOBAL_ERRSIM_POINT_DEF(751, EN_SESSION_LEAK_COUNT_THRESHOLD, "used to control the threshold of report session leak ERROR");
GLOBAL_ERRSIM_POINT_DEF(760, EN_DISABLE_TABLET_MINOR_MERGE, "used to stop scheduling minor merge");
GLOBAL_ERRSIM_POINT_DEF(800, EN_END_PARTICIPANT, "");

// compaction 801 - 899

// compaction 801 - 899
//LS Migration Related 900 - 1000
GLOBAL_ERRSIM_POINT_DEF(900, EN_INITIAL_MIGRATION_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(901, EN_START_MIGRATION_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(902, EN_SYS_TABLETS_MIGRATION_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(903, EN_DATA_TABLETS_MIGRATION_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(904, EN_TABLET_GROUP_MIGRATION_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(905, EN_TABLET_MIGRATION_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(906, EN_MIGRATION_FINISH_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(907, EN_MIGRATION_READ_REMOTE_MACRO_BLOCK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(908, EN_MIGRATION_ENABLE_LOG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(909, EN_MIGRATION_ENABLE_VOTE_RETRY, "");
GLOBAL_ERRSIM_POINT_DEF(910, EN_MIGRATION_ENABLE_VOTE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(911, EN_MIGRATION_COPY_MACRO_BLOCK_NUM, "");
GLOBAL_ERRSIM_POINT_DEF(912, EN_FINISH_TABLET_GROUP_RESTORE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(913, EN_MIGRATION_ONLINE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(914, EN_MIGRATION_GENERATE_SYS_TABLETS_DAG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(915, EN_COPY_MAJOR_SNAPSHOT_VERSION, "");
GLOBAL_ERRSIM_POINT_DEF(916, EN_TABLET_MIGRATION_DAG_INNER_RETRY, "");
GLOBAL_ERRSIM_POINT_DEF(917, EN_LS_REBUILD_PREPARE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(918, EN_TABLET_GC_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(919, EN_UPDATE_TABLET_HA_STATUS_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(920, EN_GENERATE_REBUILD_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(921, EN_CHECK_TRANSFER_TASK_EXSIT, "");
GLOBAL_ERRSIM_POINT_DEF(922, EN_TABLET_EMPTY_SHELL_TASK_FAILED, "");

// Log Archive and Restore 1001 - 1100
GLOBAL_ERRSIM_POINT_DEF(1001, EN_START_ARCHIVE_LOG_GAP, "");
GLOBAL_ERRSIM_POINT_DEF(1002, EN_RESTORE_LOG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1003, EN_RESTORE_LOG_FROM_SOURCE_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1004, EN_BACKUP_MULTIPLE_MACRO_BLOCK, "");
GLOBAL_ERRSIM_POINT_DEF(1005, EN_RESTORE_FETCH_TABLET_INFO, "");
GLOBAL_ERRSIM_POINT_DEF(1006, EN_RESTORE_COPY_MACRO_BLOCK_NUM, "");
GLOBAL_ERRSIM_POINT_DEF(1007, EN_ENABLE_ACTIVATE_AFTER_QUICK_RESTORE_FINISH, "");

// START OF STORAGE HA - 1101 - 2000
GLOBAL_ERRSIM_POINT_DEF(1101, EN_BACKUP_META_REPORT_RESULT_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1102, EN_RESTORE_LS_INIT_PARAM_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1103, EN_RESTORE_TABLET_INIT_PARAM_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1104, EN_ADD_BACKUP_META_DAG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1105, EN_ADD_BACKUP_DATA_DAG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1106, EN_ADD_BACKUP_BUILD_INDEX_DAG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1107, EN_ADD_BACKUP_PREPARE_DAG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1108, EN_ADD_BACKUP_FINISH_DAG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1109, EN_ADD_BACKUP_PREFETCH_DAG_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1110, EN_BACKUP_PERSIST_SET_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1111, EN_BACKUP_READ_MACRO_BLOCK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1112, EN_FETCH_TABLE_INFO_RPC, "");
GLOBAL_ERRSIM_POINT_DEF(1113, EN_RESTORE_TABLET_TASK_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1114, EN_INSERT_USER_RECOVER_JOB_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1115, EN_INSERT_AUX_TENANT_RESTORE_JOB_FAILED, "");
GLOBAL_ERRSIM_POINT_DEF(1116, EN_RESTORE_CREATE_LS_FAILED, "");
// END OF STORAGE HA - 1101 - 2000

// sql parameterization 1170-1180
GLOBAL_ERRSIM_POINT_DEF(1170, EN_SQL_PARAM_FP_NP_NOT_SAME_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(1171, EN_FLUSH_PC_NOT_CLEANUP_LEAK_MEM_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(1172, EN_PC_NOT_SWALLOW_ERROR, "");
// END OF sql parameterization 1170-1180

// session info verification
// The types are used for error verification
GLOBAL_ERRSIM_POINT_DEF(1180, EN_SESS_INFO_VERI_SYS_VAR_ERROR, "Used for session self-verification");
GLOBAL_ERRSIM_POINT_DEF(1181, EN_SESS_INFO_VERI_APP_INFO_ERROR, "Used for session self-verification");
GLOBAL_ERRSIM_POINT_DEF(1182, EN_SESS_INFO_VERI_APP_CTX_ERROR, "Used for session self-verification");
GLOBAL_ERRSIM_POINT_DEF(1183, EN_SESS_INFO_VERI_CLIENT_ID_ERROR, "Used for session self-verification");
GLOBAL_ERRSIM_POINT_DEF(1184, EN_SESS_INFO_VERI_CONTROL_INFO_ERROR, "Used for session self-verification");
GLOBAL_ERRSIM_POINT_DEF(1185, EN_SESS_INFO_VERI_TXN_EXTRA_INFO_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(1186, EN_SESS_POOL_MGR_CTRL, "Used for session pool use");
// session info diagnosis control
GLOBAL_ERRSIM_POINT_DEF(1187, EN_SESS_INFO_DIAGNOSIS_CONTROL, "");
GLOBAL_ERRSIM_POINT_DEF(1188, EN_SESS_CLEAN_KILL_MAP_TIME, "Used to clean up kill session map time control");
      // sql audit background thread stuck
GLOBAL_ERRSIM_POINT_DEF(1189, EN_SQL_AUDIT_RELEASE_BACK_THREAD_STUCK, "");
GLOBAL_ERRSIM_POINT_DEF(1190, EN_SQL_AUDIT_CONSTRUCT_BACK_THREAD_STUCK, "");
GLOBAL_ERRSIM_POINT_DEF(1200, EN_ENABLE_NEWSORT_FORCE, "");

// Transaction // 2001 - 2100
// Transaction free route
GLOBAL_ERRSIM_POINT_DEF(2001, EN_TX_FREE_ROUTE_UPDATE_STATE_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(2002, EN_TX_FREE_ROUTE_ENCODE_STATE_ERROR, "");
GLOBAL_ERRSIM_POINT_DEF(2003, EN_TX_FREE_ROUTE_STATE_SIZE, "");
// Transaction common
GLOBAL_ERRSIM_POINT_DEF(2011, EN_TX_RESULT_INCOMPLETE, "");
GLOBAL_ERRSIM_POINT_DEF(2013, EN_CHECK_TX_CTX_LOCK, "");
GLOBAL_ERRSIM_POINT_DEF(2022, EN_THREAD_HANG, "");

GLOBAL_ERRSIM_POINT_DEF(2100, EN_ENABLE_SET_TRACE_CONTROL_INFO, "");
GLOBAL_ERRSIM_POINT_DEF(2101, EN_CHEN, "");
GLOBAL_ERRSIM_POINT_DEF(2102, EN_ENABLE_TABLE_LOCK, "");
GLOBAL_ERRSIM_POINT_DEF(2103, EN_ENABLE_ROWKEY_CONFLICT_CHECK, "");
GLOBAL_ERRSIM_POINT_DEF(2104, EN_ENABLE_ORA_DECINT_CONST, "wether to parse constant numerics as ObDecimalIntType in orace mode");
GLOBAL_ERRSIM_POINT_DEF(2105, EN_ENABLE_CLEAN_INTERM_RES, "Used to control whether interm results are cleaned up in exceptional circumstances.");
GLOBAL_ERRSIM_POINT_DEF(2106, EN_UNIQ_TASK_QUEUE_GET_GROUP_FAIL, "");

GLOBAL_ERRSIM_POINT_DEF(2200, EN_DISABLE_VEC_SORT, "Used to control whether to turn off the vectorization 2.0 sort operator. It is turned on by default.");
GLOBAL_ERRSIM_POINT_DEF(2201, EN_DISABLE_VEC_HASH_DISTINCT, "Used to control whether to turn off the vectorization 2.0 hash distinct operator. It is turned on by default.");
GLOBAL_ERRSIM_POINT_DEF(2202, EN_DISABLE_VEC_HASH_JOIN, "Used to control whether to turn off the vectorization 2.0 when use Hash Join Operator");
GLOBAL_ERRSIM_POINT_DEF(2203, EN_DISABLE_VEC_HASH_GROUP_BY, "Used to control whether to turn off the vectorization 2.0 when use Hash Group By Operator");
GLOBAL_ERRSIM_POINT_DEF(2204, EN_DISABLE_VEC_SCALAR_GROUP_BY, "wether to use scalar groupby operator of vectorization 2.0");
GLOBAL_ERRSIM_POINT_DEF(2205, EN_DTL_OPTION, "Control DTL Vectorization 2.0 format");
GLOBAL_ERRSIM_POINT_DEF(2206, EN_ENABLE_RANDOM_BATCH_SIZE, "Used to random batch size in vectorization");
GLOBAL_ERRSIM_POINT_DEF(2207, EN_ENABLE_VECTOR_CAST, "wether to use casting functions of vectorization 2.0");
GLOBAL_ERRSIM_POINT_DEF(2208, EN_DISABLE_SORTKEY_SEPARATELY, "Used to control whether to turn off the separate storage of sort keys and addon fields. It is enabled by default.");
GLOBAL_ERRSIM_POINT_DEF(2209, EN_ENABLE_VECTOR_IN, "Used to control whether the capability for in-expr vectorization 2.0 is enabled.");
GLOBAL_ERRSIM_POINT_DEF(2210, EN_SQL_MEMORY_MRG_OPTION, "Control automatic memory management global bound size");
GLOBAL_ERRSIM_POINT_DEF(2211, EN_ENABLE_RANDOM_TSC, "wether to randomize batch_size & skips of table scan's output ");
GLOBAL_ERRSIM_POINT_DEF(2212, EN_LOCK_CONFLICT_RETRY_THEN_REROUTE, "force reroute sql when lock conflict and retry a few times");

// WR && ASH
GLOBAL_ERRSIM_POINT_DEF(2301, EN_CLOSE_ASH, "");
GLOBAL_ERRSIM_POINT_DEF(2302, EN_DISABLE_HASH_BASE_DISTINCT, "");
GLOBAL_ERRSIM_POINT_DEF(2303, EN_DISABLE_VEC_WINDOW_FUNCTION, "Disable window function operator of vectorization 2.0");
GLOBAL_ERRSIM_POINT_DEF(2304, EN_FORCE_WINFUNC_STORE_DUMP, "Force to dump row store of window function operator");
GLOBAL_ERRSIM_POINT_DEF(2305, EN_TRACEPOINT_TEST, "For testing new versions of tracepoint");

GLOBAL_ERRSIM_POINT_DEF(2306, EN_DISABLE_VEC_MERGE_DISTINCT, "Used to control whether to turn off the vectorization 2.0 merge distinct operator. It is turned on by default.");
// force dump
GLOBAL_ERRSIM_POINT_DEF(2400, EN_SQL_FORCE_DUMP, "For testing force dump once");
GLOBAL_ERRSIM_POINT_DEF(2401, EN_TEST_FOR_HASH_UNION, "Used to control whether to turn off the vectorization 2.0 hash set operator. It is turned on by default.");

// Protocol begin 2450 - 2500
// pr-exec protocol
GLOBAL_ERRSIM_POINT_DEF(2450, COM_STMT_PREXECUTE_PREPARE_ERROR, "inject error at prepare stage for pr-exec protocol");
GLOBAL_ERRSIM_POINT_DEF(2451, COM_STMT_PREXECUTE_PS_CURSOR_OPEN_ERROR, "inject error at cursor open stage for pr-exec protocol");
GLOBAL_ERRSIM_POINT_DEF(2452, COM_STMT_PREXECUTE_EXECUTE_ERROR, "inject error at execute stage for pr-exec protocol");
GLOBAL_ERRSIM_POINT_DEF(2453, EN_ENABLE_NEW_RESULT_META_DATA, "For testing enable new result meta data, off by default");

// Protocol end

GLOBAL_ERRSIM_POINT_DEF(2501, EN_CHECK_SORT_CMP, "Used to check the legality of the compare method for std::sort");

#endif /*GLOBAL_ERRSIM_POINT_DEF*/
