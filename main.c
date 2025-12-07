#include "edit_distance.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define BENCHMARK_LEN 1000000

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void generate_random_string(char *str, size_t len) {
    for (size_t i = 0; i < len; i++) {
        str[i] = 'A' + (rand() % 26);
    }
    str[len] = '\0';
}

int main() {
    printf("Edit Distance Benchmark Test\n");
    
    srand(time(NULL));
    
    char *str1 = malloc(BENCHMARK_LEN + 1);
    char *str2 = malloc(BENCHMARK_LEN + 1);
    
    
    generate_random_string(str1, BENCHMARK_LEN);
    generate_random_string(str2, BENCHMARK_LEN);
    
    printf("Computing edit distance...\n");
    double start = get_time();
    int distance = cse2421_edit_distance(str1, str2, BENCHMARK_LEN);
    double end = get_time();
    
    double elapsed = end - start;
    
    printf("\n");
    printf("Results:\n\n");

    printf("String length: %d\n", BENCHMARK_LEN);
    printf("Edit distance: %d\n", distance);
    printf("Execution time: %.3f seconds\n", elapsed);
    
    free(str1);
    free(str2);
    
    return 0;
}