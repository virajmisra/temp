lab7: edit_distance.c edit_distance.h test_edit_distance.c main.c
	gcc -O3 -mavx2 -pthread -Wall -o lab7 edit_distance.c test_edit_distance.c main.c