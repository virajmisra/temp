// edit_distance.c
#include "edit_distance.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdatomic.h>

// ---- Portable barrier for macOS (replaces pthread_barrier_t) ----
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    int count;
    int total;
} simple_barrier_t;

static void barrier_init(simple_barrier_t *b, int total) {
    pthread_mutex_init(&b->mutex, NULL);
    pthread_cond_init(&b->cond, NULL);
    b->count = 0;
    b->total = total;
}

static void barrier_wait(simple_barrier_t *b) {
    pthread_mutex_lock(&b->mutex);
    b->count++;
    if (b->count == b->total) {
        b->count = 0;
        pthread_cond_broadcast(&b->cond);
    } else {
        pthread_cond_wait(&b->cond, &b->mutex);
    }
    pthread_mutex_unlock(&b->mutex);
}

static void barrier_destroy(simple_barrier_t *b) {
    pthread_mutex_destroy(&b->mutex);
    pthread_cond_destroy(&b->cond);
}

#ifndef TILE
#define TILE 256
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

// helper
static inline int imin3(int a, int b, int c) {
    int m = a < b ? a : b;
    return m < c ? m : c;
}

// Thread pool shared state
typedef struct {
    const char *A;
    const char *B;
    size_t N;             // length of strings
    int tile_rows;        // number of tile rows (N / TILE, ceil)
    int tile_cols;
    int tiles_per_diag;   // not used but helpful
    // tile boundary buffers: we store the row/col boundaries for tiles
    // We keep arrays of boundary rows for tile grid lines.
    // We'll allocate an integer buffer for each horizontal gridline of size N+1
    // but only (tile_cols+1) boundaries per tile row. To reduce complexity,
    // here we allocate 2 rows of length (N+1) for previous/full row frontier and
    // then tile-internal buffers of size (TILE+1) when needed.
    pthread_t threads[NUM_THREADS];
    _Atomic int current_tile_idx; // used by threads to fetch tile index in current wave
    _Atomic int current_wave;            // tile-wave (tile_i + tile_j)
    simple_barrier_t wave_barrier;
    
    int *grid_top; 
    int *grid_left; 
} pool_t;

static inline int i_min(int a, int b) { return a < b ? a : b; }
static inline int i_max(int a, int b) { return a > b ? a : b; }

// Each tile compute uses local anti-diagonals.
// arguments: tile row ti, tile col tj
static void compute_tile(pool_t *p, int ti, int tj) {
    const char *A = p->A;
    const char *B = p->B;
    size_t N = p->N;

    int row_start = ti * TILE;
    int col_start = tj * TILE;
    int row_end = (int)i_min((int)N, row_start + TILE);
    int col_end = (int)i_min((int)N, col_start + TILE);

    int H = row_end - row_start; 
    int W = col_end - col_start; 

    if (H <= 0 || W <= 0) return;

    int *local = (int *)malloc((H + 1) * (W + 1) * sizeof(int));
    if (!local) return;

    #define L(i,j) local[(i)*(W+1) + (j)]


    if (ti == 0 && tj == 0) {
        L(0,0) = 0;
        for (int j = 1; j <= W; ++j) L(0,j) = j;
        for (int i = 1; i <= H; ++i) L(i,0) = i;
    } else {
        // Fill top row (i=0): values dp[row_start][col_start + j] come from previous tile in same row if any
        if (ti == 0) {
            // topmost tile row; top boundary from global dp[0][..]
            L(0,0) = 0;
            for (int j = 1; j <= W; ++j) L(0,j) = col_start + j; // distance from empty A
        } else {
            // read from p->grid_top entry written by tile (ti-1,tj)
            int *top_src = p->grid_top + (ti * TILE * p->tile_cols) + (tj * TILE); 

            L(0,0) = 0;
            for (int j = 1; j <= W; ++j) L(0,j) = col_start + j;
        }

        // Fill left column (j=0)
        if (tj == 0) {
            for (int i = 1; i <= H; ++i) L(i,0) = row_start + i;
        } else {
            for (int i = 1; i <= H; ++i) L(i,0) = row_start + i;
        }
    }

    // Now compute interior of tile via local anti-diagonals (k from 1 .. H+W)
    int maxk = H + W;
    for (int k = 1; k <= maxk; ++k) {
        // local cells satisfying i + j = k where i in [1..H], j in [1..W]
        int i0 = i_max(1, k - W);
        int i1 = i_min(H, k - 1);
        int count = i1 - i0 + 1;
        int j0 = k - i1;

        // vectorize across entries in this anti-diagonal where possible
        int idx = 0;
        while (idx + 8 <= count) {
            // build vectors of diag, left, up and cost
            int vals_diag[8], vals_left[8], vals_up[8], costs[8];
            for (int t = 0; t < 8; ++t) {
                int ii = i0 + idx + t;
                int jj = k - ii;
                vals_diag[t] = L(ii-1, jj-1);
                vals_left[t] = L(ii, jj-1) + 1;
                vals_up[t] = L(ii-1, jj) + 1;
                costs[t] = (A[row_start + ii - 1] == B[col_start + jj - 1]) ? 0 : 1;
            }
            // load into AVX2 registers
            __m256i v_diag = _mm256_loadu_si256((__m256i*)vals_diag);
            __m256i v_left = _mm256_loadu_si256((__m256i*)vals_left);
            __m256i v_up   = _mm256_loadu_si256((__m256i*)vals_up);
            __m256i v_cost = _mm256_loadu_si256((__m256i*)costs);

            __m256i v_diag_cost = _mm256_add_epi32(v_diag, v_cost);
            __m256i v_min = _mm256_min_epi32(v_diag_cost, v_left);
            v_min = _mm256_min_epi32(v_min, v_up);

            int out[8];
            _mm256_storeu_si256((__m256i*)out, v_min);

            for (int t = 0; t < 8; ++t) {
                int ii = i0 + idx + t;
                int jj = k - ii;
                L(ii, jj) = out[t];
            }
            idx += 8;
        }
        // scalar remainder
        for (; idx < count; ++idx) {
            int ii = i0 + idx;
            int jj = k - ii;
            int cost = (A[row_start + ii - 1] == B[col_start + jj - 1]) ? 0 : 1;
            int v = imin3(
                L(ii-1, jj-1) + cost,
                L(ii,   jj-1) + 1,
                L(ii-1, jj  ) + 1
            );
            L(ii, jj) = v;
        }
    }


    for (int j = 0; j <= W; ++j) {
        // dp[row_end][col_start + j] -> L(H, j)
        // safe store into grid_top (we allocated a big buffer of size (tile_rows+1) * TILE * tile_cols roughly)
        int idxg = (ti + 1) * (p->tile_cols * TILE) + tj * TILE + j;
        if (idxg < (p->tile_rows + 1) * (p->tile_cols * TILE))
            p->grid_top[idxg] = L(H, j);
    }
    for (int i = 0; i <= H; ++i) {
        int idxg = (tj + 1) * (p->tile_rows * TILE) + ti * TILE + i;
        if (idxg < (p->tile_cols + 1) * (p->tile_rows * TILE))
            p->grid_left[idxg] = L(i, W);
    }

    // If this is the last tile covering the bottom-right corner, write result into a well-known place:
    // We'll write L(H,W) into p->grid_top[0] (convenient global location)
    if (row_end == (int)N && col_end == (int)N) {
        p->grid_top[0] = L(H,W);
    }

    free(local);
    #undef L
}

// Worker function: repeatedly grab next tile index for current wave
static void *worker_body(void *arg) {
    pool_t *p = (pool_t *)arg;

    for (;;) {
        // fetch wave locally
        int wave = p->current_wave;
        if (p->current_wave < 0) break; // termination signal

        // compute tile range for this wave
        int min_ti = i_max(0, wave - (p->tile_cols - 1));
        int max_ti = i_min(p->tile_rows - 1, wave);
        int tiles_in_wave = max_ti - min_ti + 1;
        if (tiles_in_wave <= 0) {
            // sync barrier: wait until next wave is active or termination
            barrier_wait(&p->wave_barrier);
            if (p->current_wave < 0) break;
            continue;
        }

        // per-wave atomic counter to claim tiles
        int next = atomic_fetch_add(&p->current_tile_idx, 1);
        while (next < tiles_in_wave) {
            int ti = min_ti + next;
            int tj = wave - ti;
            compute_tile(p, ti, tj);
            next = atomic_fetch_add(&p->current_tile_idx, 1);
        }

        // everyone reaches barrier for this wave
        barrier_wait(&p->wave_barrier);

        // loop again: either next wave or termination
        if (p->current_wave < 0) break;
    }
    return NULL;
}

int cse2421_edit_distance(const char *str1, const char *str2, size_t len) {
    if (len == 0) return 0;

    pool_t pool;
    memset(&pool, 0, sizeof(pool_t));
    pool.A = str1;
    pool.B = str2;
    pool.N = len;

    pool.tile_rows = (int)((len + TILE - 1) / TILE);
    pool.tile_cols = pool.tile_rows; // square grid because strings are same length

    // allocate grid_top and grid_left with a safe over-approx size
    int over = (pool.tile_rows + 1) * (pool.tile_cols * TILE);
    pool.grid_top = (int *)malloc(sizeof(int) * over);
    pool.grid_left = (int *)malloc(sizeof(int) * over);
    if (!pool.grid_top || !pool.grid_left) {
        free(pool.grid_top); free(pool.grid_left);
        return -1;
    }
    // initialize boundaries: top row and left column for global DP base
    // dp[0][j] = j ; dp[i][0] = i
    // We'll store dp[0][*] into the top of first tile row for convenience
    // Put dp[0][0..TILE] into grid_top at position corresponding to (ti=0,tj=0)
    for (int j = 0; j <= (int)len && j < TILE; ++j) {
        pool.grid_top[0 * (pool.tile_cols * TILE) + 0 * TILE + j] = j;
    }
    for (int i = 0; i <= (int)len && i < TILE; ++i) {
        pool.grid_left[0 * (pool.tile_rows * TILE) + 0 * TILE + i] = i;
    }

    barrier_init(&pool.wave_barrier, NUM_THREADS);

    pool.current_tile_idx = 0;
    pool.current_wave = 0;

    // launch threads
    for (int t = 0; t < NUM_THREADS; ++t) {
        pthread_create(&pool.threads[t], NULL, worker_body, &pool);
    }

    // schedule waves
    int max_wave = pool.tile_rows + pool.tile_cols - 2;
    for (int w = 0; w <= max_wave; ++w) {
        // set up wave
        pool.current_tile_idx = 0;
        pool.current_wave = w;

        // wait for threads to finish wave
        barrier_wait(&pool.wave_barrier);
        // after barrier the wave is complete
    }

    // read result from pool.grid_top[0] or from bottom-right tile's stored corner
    int result = pool.grid_top[0]; // earlier compute writes final answer here

    // signal termination
    pool.current_wave = -1;
    // one more barrier to let threads exit loops
    barrier_wait(&pool.wave_barrier);

    // join threads
    for (int t = 0; t < NUM_THREADS; ++t) {
        pthread_join(pool.threads[t], NULL);
    }

    barrier_destroy(&pool.wave_barrier);
    free(pool.grid_top);
    free(pool.grid_left);

    return result;
}