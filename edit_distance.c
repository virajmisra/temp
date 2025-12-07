#include "edit_distance.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <stdint.h>


#define TILE_SIZE 256
#define NUM_THREADS 8
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
    int tile_row;
    int tile_col;
    int wave_id;
} tile_work_t;

typedef struct {
    const char *str1;
    const char *str2;
    size_t len;
    int *prev_row;
    int *curr_row;
    int num_tile_rows;
    int num_tile_cols;
    

    tile_work_t *work_queue;
    int queue_size;
    int queue_head;
    pthread_mutex_t queue_mutex;
    pthread_cond_t work_available;
    pthread_cond_t wave_complete;
    int active_workers;
    int tiles_completed;
    int tiles_in_wave;
    int done;
} shared_state_t;

static inline int min3(int a, int b, int c) {
    return min(min(a, b), c);
}

static void compute_tile_avx2(shared_state_t *state, int tile_row, int tile_col) {
    size_t start_i = tile_row * TILE_SIZE;
    size_t start_j = tile_col * TILE_SIZE;
    size_t end_i = min(start_i + TILE_SIZE, state->len);
    size_t end_j = min(start_j + TILE_SIZE, state->len);
    
    for (size_t i = start_i; i < end_i; i++) {
        size_t j = start_j;
        
        for (; j + 8 <= end_j; j += 8) {
            __m256i left = _mm256_loadu_si256((__m256i*)&state->curr_row[j]);
            __m256i diag = _mm256_loadu_si256((__m256i*)&state->prev_row[j]);
            __m256i up = _mm256_loadu_si256((__m256i*)&state->prev_row[j + 1]);
            
            
            __m256i chars1 = _mm256_set1_epi32(state->str1[i]);
            __m256i chars2 = _mm256_set_epi32(
                state->str2[j + 7], state->str2[j + 6],
                state->str2[j + 5], state->str2[j + 4],
                state->str2[j + 3], state->str2[j + 2],
                state->str2[j + 1], state->str2[j + 0]
            );
            __m256i match = _mm256_cmpeq_epi32(chars1, chars2);
            __m256i cost = _mm256_andnot_si256(match, _mm256_set1_epi32(1));
            
            
            __m256i diag_cost = _mm256_add_epi32(diag, cost);
            
     
            __m256i left_cost = _mm256_add_epi32(left, _mm256_set1_epi32(1));
            
      
            __m256i up_cost = _mm256_add_epi32(up, _mm256_set1_epi32(1));
            
           
            __m256i min_val = _mm256_min_epi32(diag_cost, left_cost);
            
            
            min_val = _mm256_min_epi32(min_val, up_cost);
            
            _mm256_storeu_si256((__m256i*)&state->curr_row[j + 1], min_val);
        }
        
     
        for (; j < end_j; j++) {
            int cost = (state->str1[i] == state->str2[j]) ? 0 : 1;
            int val = min3(
                state->prev_row[j] + cost,
                state->curr_row[j] + 1,
                state->prev_row[j + 1] + 1
            );
            state->curr_row[j + 1] = val;
        }
        
       
        int *tmp = state->prev_row;
        state->prev_row = state->curr_row;
        state->curr_row = tmp;
    }
}


static void* worker_thread(void *arg) {
    shared_state_t *state = (shared_state_t*)arg;
    
    while (1) {
        pthread_mutex_lock(&state->queue_mutex);
        
     
        while (state->queue_head >= state->queue_size && !state->done) {
            pthread_cond_wait(&state->work_available, &state->queue_mutex);
        }
        
        if (state->done) {
            pthread_mutex_unlock(&state->queue_mutex);
            break;
        }
        

        tile_work_t work = state->work_queue[state->queue_head++];
        state->active_workers++;
        pthread_mutex_unlock(&state->queue_mutex);
        

        compute_tile_avx2(state, work.tile_row, work.tile_col);
        

        pthread_mutex_lock(&state->queue_mutex);
        state->active_workers--;
        state->tiles_completed++;
        
        if (state->tiles_completed == state->tiles_in_wave) {
            pthread_cond_signal(&state->wave_complete);
        }
        pthread_mutex_unlock(&state->queue_mutex);
    }
    
    return NULL;
}


int cse2421_edit_distance(const char *str1, const char *str2, size_t len) {
    if (len == 0) return 0;
    

    int *prev_row = malloc((len + 1) * sizeof(int));
    int *curr_row = malloc((len + 1) * sizeof(int));
    

    for (size_t i = 0; i <= len; i++) {
        prev_row[i] = i;
    }

    shared_state_t state;
    state.str1 = str1;
    state.str2 = str2;
    state.len = len;
    state.prev_row = prev_row;
    state.curr_row = curr_row;
    state.num_tile_rows = (len + TILE_SIZE - 1) / TILE_SIZE;
    state.num_tile_cols = (len + TILE_SIZE - 1) / TILE_SIZE;
    state.done = 0;
    
    pthread_mutex_init(&state.queue_mutex, NULL);
    pthread_cond_init(&state.work_available, NULL);
    pthread_cond_init(&state.wave_complete, NULL);
    

    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker_thread, &state);
    }
    
 
    int max_waves = state.num_tile_rows + state.num_tile_cols - 1;
    state.work_queue = malloc(max(state.num_tile_rows, state.num_tile_cols) * sizeof(tile_work_t));
    
    for (int wave = 0; wave < max_waves; wave++) {

        state.queue_head = 0;
        state.queue_size = 0;
        state.tiles_completed = 0;
        state.active_workers = 0;
        
        for (int tile_row = 0; tile_row < state.num_tile_rows; tile_row++) {
            int tile_col = wave - tile_row;
            if (tile_col >= 0 && tile_col < state.num_tile_cols) {
                state.work_queue[state.queue_size].tile_row = tile_row;
                state.work_queue[state.queue_size].tile_col = tile_col;
                state.work_queue[state.queue_size].wave_id = wave;
                state.queue_size++;
            }
        }
        
        state.tiles_in_wave = state.queue_size;
        

        pthread_mutex_lock(&state.queue_mutex);
        pthread_cond_broadcast(&state.work_available);
        pthread_mutex_unlock(&state.queue_mutex);
        

        pthread_mutex_lock(&state.queue_mutex);
        while (state.tiles_completed < state.tiles_in_wave) {
            pthread_cond_wait(&state.wave_complete, &state.queue_mutex);
        }
        pthread_mutex_unlock(&state.queue_mutex);
    }
    
    pthread_mutex_lock(&state.queue_mutex);
    state.done = 1;
    pthread_cond_broadcast(&state.work_available);
    pthread_mutex_unlock(&state.queue_mutex);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    int result = state.prev_row[len];
 
    free(prev_row);
    free(curr_row);
    free(state.work_queue);
    pthread_mutex_destroy(&state.queue_mutex);
    pthread_cond_destroy(&state.work_available);
    pthread_cond_destroy(&state.wave_complete);
    
    return result;
}