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


// [0, 100) for inner group
// CGID_DEF(group_name, group_id[, flags=DEFAULT][, worker_concurrency=1])
// example: CGID_DEF(OBCG_EXAMPLE1, 1, CRITICAL)
//          CGID_DEF(OBCG_EXAMPLE2, 2, DEFAULT, 4)
// flags option:
//     DEFAULT. No flags.
//     CRITICAL. If a group is not critical, the thread num of it can be set to 0 when idle.
CGID_DEF(OBCG_DEFAULT, 0)
CGID_DEF(OBCG_CLOG, 1)
CGID_DEF(OBCG_ELECTION, 2, CRITICAL)
CGID_DEF(OBCG_ID_SERVICE, 5, CRITICAL)
CGID_DEF(OBCG_ID_SQL_REQ_LEVEL1, 6, DEFAULT, 4)
CGID_DEF(OBCG_ID_SQL_REQ_LEVEL2, 7, DEFAULT, 4)
CGID_DEF(OBCG_ID_SQL_REQ_LEVEL3, 8, DEFAULT, 4)
CGID_DEF(OBCG_DETECT_RS, 9, CRITICAL)
CGID_DEF(OBCG_LOC_CACHE, 10, CRITICAL)
CGID_DEF(OBCG_SQL_NIO, 11)
CGID_DEF(OBCG_MYSQL_LOGIN, 12)
CGID_DEF(OBCG_CDCSERVICE, 13)
CGID_DEF(OBCG_DIAG_TENANT, 14)
CGID_DEF(OBCG_WR, 15)
CGID_DEF(OBCG_TRANSFER, 16)
CGID_DEF(OBCG_STORAGE_STREAM, 17)
CGID_DEF(OBCG_DBA_COMMAND, 18, DEFAULT, 0)
CGID_DEF(OBCG_STORAGE, 19)
CGID_DEF(OBCG_LOCK, 20)
CGID_DEF(OBCG_UNLOCK, 21)
CGID_DEF(OBCG_DIRECT_LOAD_HIGH_PRIO, 22)
CGID_DEF(OBCG_DDL, 23)
CGID_DEF(OBCG_USER_LOCK, 24)
CGID_DEF(OBCG_HB_SERVICE, 25, CRITICAL)
CGID_DEF(OBCG_OLAP_ASYNC_JOB, 26)


// 100 for CG_LQ
CGID_DEF(OBCG_LQ, 100)
