#include "edit_distance.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

void test_identical_strings() {
    const char *s = "hello";
    int dist = cse2421_edit_distance(s, s, strlen(s));
    assert(dist == 0);
}

void test_empty_strings() {
    const char *s = "";
    int dist = cse2421_edit_distance(s, s, 0);
    assert(dist == 0);
}

void test_one_substitution() {
    const char *s1 = "kitten";
    const char *s2 = "sitten";
    int dist = cse2421_edit_distance(s1, s2, strlen(s1));
    assert(dist == 1);
}

void test_kitten_sitting() {
    const char *s1 = "kitten";
    const char *s2 = "sitting";
    int dist = cse2421_edit_distance(s1, s2, 7);
    assert(dist == 3);
}

void test_completely_different() {
    const char *s1 = "abc";
    const char *s2 = "xyz";
    int dist = cse2421_edit_distance(s1, s2, 3);
    assert(dist == 3);
}

void test_single_char() {
    const char *s1 = "a";
    const char *s2 = "b";
    int dist = cse2421_edit_distance(s1, s2, 1);
    assert(dist == 1);
    
    const char *s3 = "a";
    const char *s4 = "a";
    dist = cse2421_edit_distance(s3, s4, 1);
    assert(dist == 0);
}

void test_longer_string() {
    const char *s1 = "abcdefghijklmnop";
    const char *s2 = "abcdefghijklmnop";
    int dist = cse2421_edit_distance(s1, s2, strlen(s1));
    assert(dist == 0);
    
    const char *s3 = "abcdefghijklmnop";
    const char *s4 = "ABCDEFGHIJKLMNOP";
    dist = cse2421_edit_distance(s3, s4, strlen(s3));
    assert(dist == 16);
}

void test_tile_boundary() {
    char s1[300];
    char s2[300];
    
    for (int i = 0; i < 299; i++) {
        s1[i] = 'A';
        s2[i] = 'A';
    }
    s1[299] = '\0';
    s2[299] = '\0';
    
    int dist = cse2421_edit_distance(s1, s2, 299);
    assert(dist == 0);
    
    s2[150] = 'B';
    dist = cse2421_edit_distance(s1, s2, 299);
    assert(dist == 1);
}

int main() {
    printf("Testing edit distance\n\n");
    
    test_empty_strings();
    test_identical_strings();
    test_one_substitution();
    test_kitten_sitting();
    test_completely_different();
    test_single_char();
    test_longer_string();
    test_tile_boundary();
    
    printf("\nAll tests passed\n");
    return 0;
}