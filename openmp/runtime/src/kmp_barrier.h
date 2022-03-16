/*
 * kmp_barrier.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_BARRIER_H
#define KMP_BARRIER_H

#include "kmp.h"
#include "kmp_itt.h"
#include "kmp_i18n.h"

#include <fcntl.h>
#include "kmp_safe_c_api.h"

#if KMP_HAVE_XMMINTRIN_H && KMP_HAVE__MM_MALLOC
#include <xmmintrin.h>
#define KMP_ALIGNED_ALLOCATE(size, alignment) _mm_malloc(size, alignment)
#define KMP_ALIGNED_FREE(ptr) _mm_free(ptr)
#elif KMP_HAVE_ALIGNED_ALLOC
#define KMP_ALIGNED_ALLOCATE(size, alignment) aligned_alloc(alignment, size)
#define KMP_ALIGNED_FREE(ptr) free(ptr)
#elif KMP_HAVE_POSIX_MEMALIGN
static inline void *KMP_ALIGNED_ALLOCATE(size_t size, size_t alignment) {
  void *ptr;
  int n = posix_memalign(&ptr, alignment, size);
  if (n != 0) {
    if (ptr)
      free(ptr);
    return nullptr;
  }
  return ptr;
}
#define KMP_ALIGNED_FREE(ptr) free(ptr)
#elif KMP_HAVE__ALIGNED_MALLOC
#include <malloc.h>
#define KMP_ALIGNED_ALLOCATE(size, alignment) _aligned_malloc(size, alignment)
#define KMP_ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#define KMP_ALIGNED_ALLOCATE(size, alignment) KMP_INTERNAL_MALLOC(size)
#define KMP_ALIGNED_FREE(ptr) KMP_INTERNAL_FREE(ptr)
#endif

// Use four cache lines: MLC tends to prefetch the next or previous cache line
// creating a possible fake conflict between cores, so this is the only way to
// guarantee that no such prefetch can happen.
#ifndef KMP_FOURLINE_ALIGN_CACHE
#define KMP_FOURLINE_ALIGN_CACHE KMP_ALIGN(4 * CACHE_LINE)
#endif

#define KMP_OPTIMIZE_FOR_REDUCTIONS 0

class distributedBarrier {
  struct flags_s {
    kmp_uint32 volatile KMP_FOURLINE_ALIGN_CACHE stillNeed;
  };

  struct go_s {
    std::atomic<kmp_uint64> KMP_FOURLINE_ALIGN_CACHE go;
  };

  struct iter_s {
    kmp_uint64 volatile KMP_FOURLINE_ALIGN_CACHE iter;
  };

  struct sleep_s {
    std::atomic<bool> KMP_FOURLINE_ALIGN_CACHE sleep;
  };

  void init(size_t nthr);
  void resize(size_t nthr);
  void computeGo(size_t n);
  void computeVarsForN(size_t n);

public:
  enum {
    MAX_ITERS = 3,
    MAX_GOS = 8,
    IDEAL_GOS = 4,
    IDEAL_CONTENTION = 16,
  };

  flags_s *flags[MAX_ITERS];
  go_s *go;
  iter_s *iter;
  sleep_s *sleep;

  size_t KMP_ALIGN_CACHE num_threads; // number of threads in barrier
  size_t KMP_ALIGN_CACHE max_threads; // size of arrays in data structure
  // number of go signals each requiring one write per iteration
  size_t KMP_ALIGN_CACHE num_gos;
  // number of groups of gos
  size_t KMP_ALIGN_CACHE num_groups;
  // threads per go signal
  size_t KMP_ALIGN_CACHE threads_per_go;
  bool KMP_ALIGN_CACHE fix_threads_per_go;
  // threads per group
  size_t KMP_ALIGN_CACHE threads_per_group;
  // number of go signals in a group
  size_t KMP_ALIGN_CACHE gos_per_group;
  void *team_icvs;

  distributedBarrier() = delete;
  ~distributedBarrier() = delete;

  // Used instead of constructor to create aligned data
  static distributedBarrier *allocate(int nThreads) {
    distributedBarrier *d = (distributedBarrier *)KMP_ALIGNED_ALLOCATE(
        sizeof(distributedBarrier), 4 * CACHE_LINE);
    if (!d) {
      KMP_FATAL(MemoryAllocFailed);
    }
    d->num_threads = 0;
    d->max_threads = 0;
    for (int i = 0; i < MAX_ITERS; ++i)
      d->flags[i] = NULL;
    d->go = NULL;
    d->iter = NULL;
    d->sleep = NULL;
    d->team_icvs = NULL;
    d->fix_threads_per_go = false;
    // calculate gos and groups ONCE on base size
    d->computeGo(nThreads);
    d->init(nThreads);
    return d;
  }

  static void deallocate(distributedBarrier *db) { KMP_ALIGNED_FREE(db); }

  void update_num_threads(size_t nthr) { init(nthr); }

  bool need_resize(size_t new_nthr) { return (new_nthr > max_threads); }
  size_t get_num_threads() { return num_threads; }
  kmp_uint64 go_release();
  void go_reset();
};

class hardBarrier {
private:
  /* group number */
  int *group;
  /* barrier blade number */
  int *bb;
  /* barrier window number on each core (indexed by tid) */
  int **barrier_window;

  int primary_core;
  int get_window_index_from_tid(int tid) { return tid % CORES_PER_GROUP; }

public:
  static constexpr const char *SYSFS_ROOT = "/sys/class/misc/hard_barrier";
  static const int BUF_SIZE = 64;
  // const value read from sysfs
  static int NUM_GROUPS;
  static int CORES_PER_GROUP;

  int num_groups;
  int *threads_in_group;
  bool is_hybrid;
  void *team_icvs;

  static hardBarrier *allocate() {
    if (NUM_GROUPS == 0) {
      // This only happens when allocating hot team in register_root().
      // At that time affinity has not been initialized and
      // we cannot determine if hard barrier can be used or not.
      // So, return NULL here to defer allocation
      return NULL;
    }

    hardBarrier *h = (hardBarrier *)KMP_ALIGNED_ALLOCATE(sizeof(hardBarrier),
                                                         4 * CACHE_LINE);
    if (!h) {
      KMP_FATAL(MemoryAllocFailed);
    }
    h->team_icvs = __kmp_allocate(sizeof(kmp_internal_control));
    h->threads_in_group = (int *)__kmp_allocate(sizeof(int) * NUM_GROUPS);
    h->group = (int *)__kmp_allocate(sizeof(int) * NUM_GROUPS);
    h->bb = (int *)__kmp_allocate(sizeof(int) * NUM_GROUPS);
    h->barrier_window = (int **)__kmp_allocate(sizeof(int *) * NUM_GROUPS);

    h->is_hybrid = false;
    h->num_groups = 0;
    for (int i = 0; i < NUM_GROUPS; i++) {
      h->group[i] = -1;
      h->bb[i] = -1;
      h->threads_in_group[i] = 0;
      h->barrier_window[i] =
          (int *)__kmp_allocate(sizeof(int) * hardBarrier::CORES_PER_GROUP);
      for (int j = 0; j < hardBarrier::CORES_PER_GROUP; j++) {
        h->barrier_window[i][j] = -1;
      }
    }
    return h;
  }
  static void deallocate(hardBarrier *h) {
    h->barrier_free();
    for (int i = 0; i < hardBarrier::NUM_GROUPS; i++) {
      __kmp_free(h->barrier_window[i]);
    }
    __kmp_free(h->barrier_window);
    __kmp_free(h->bb);
    __kmp_free(h->group);
    __kmp_free(h->threads_in_group);
    __kmp_free(h->team_icvs);
    KMP_ALIGNED_FREE(h);
  }

  hardBarrier() = delete;
  ~hardBarrier() = delete;

  int barrier_alloc(kmp_info_t *this_thr, int nthreads);
  void barrier_free();
  bool is_barrier_allocated() {
    if (num_groups > 0) {
      return true;
    }
    return false;
  };
  void get_window(kmp_info_t *this_thr, int tid);
  void sync(kmp_info_t *this_thr, int final_spin, int gtid,
            int tid USE_ITT_BUILD_ARG(void *itt_sync_obj));
  static void wakeup() {
    // wakeup threads which entered to sleep by wfe
    asm volatile("sevl" :::);
  }
  static bool system_supports_hard_barrier() {
    int ret, fd;
    char buf[BUF_SIZE];

    KMP_SNPRINTF(buf, BUF_SIZE, "%s/group0/barrier0/masks", SYSFS_ROOT);
    fd = open(buf, O_RDWR);
    if (fd < 0) {
      // module not loaded or permission denied
      return false;
    }
    close(fd);

    // check avilable barrier
    for (int group = 0;; group++) {
      KMP_SNPRINTF(buf, BUF_SIZE, "%s/group%d/available_cpus", SYSFS_ROOT,
                   group);
      fd = open(buf, O_RDONLY);
      if (fd < 0) {
        NUM_GROUPS = group;
        break;
      }

      ret = read(fd, buf, BUF_SIZE);
      KMP_ASSERT(ret > 0);
      buf[ret] = '\0';
      int start, end;
      ret = KMP_SSCANF(buf, "%d-%d", &start, &end);
      if (CORES_PER_GROUP == 0) {
        CORES_PER_GROUP = end - start + 1;
      } else {
        // XXX: Assume all group has the same number of cores for now
        KMP_ASSERT(CORES_PER_GROUP == end - start + 1);
      }
      close(fd);
    }

    KMP_ASSERT(hardBarrier::NUM_GROUPS > 0 && hardBarrier::CORES_PER_GROUP > 0);

    return true;
  }

  // below functinos also support when primary's core is not the
  // first one of the group
  bool is_group_leader(int tid) {
    return tid == 0 || ((tid + primary_core) % CORES_PER_GROUP) == 0;
  }
  int get_group_from_tid(int tid) {
    return (tid + (primary_core % CORES_PER_GROUP)) / CORES_PER_GROUP;
  }
  int get_tid_of_group_leader(int group) {
    return (group * CORES_PER_GROUP) - (primary_core % CORES_PER_GROUP);
  }
};

#endif // KMP_BARRIER_H
