#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define	CLOCKS_PER_SEC	((clock_t)1000)

float sec(clock_t clocks)
{
    return (float)clocks/CLOCKS_PER_SEC;
}

int main(int argc, char *argv[]){
	int i, j;
	int count = 100000;
	if (argc > 1)
		count = atoi(argv[1]);

	register float fa = 1024.0;
	register float fb = 1024.0;
	register float fc;
	register int ia = 1024;
	register int ib = 1024;
	register int ic;


	clock_t time = clock();
	for (i = 0; i < count; i++)
		for (j = 0; j < count; j++){
			fc = fa + fb;
		}
	float float_add = sec(clock()-time);
	printf("Float add done in %f seconds.\n", sec(float_add));


	time = clock();
	for (i = 0; i < count; i++)
		for (j = 0; j < count; j++){
			ic = ia + ib;
		}
	float int_add = sec(clock()-time);
	printf("Integer add done in %f seconds.\n", sec(int_add));

	time = clock();
	for (i = 0; i < count; i++)
		for (j = 0; j < count; j++){
			fc = fa * fb;
		}
	float float_mul = sec(clock()-time);
	printf("Float multi done in %f seconds.\n", sec(float_mul));


	time = clock();
	for (i = 0; i < count; i++)
		for (j = 0; j < count; j++){
			ic = ia * ib;
		}
	float int_mul = sec(clock()-time);
	printf("Integer multi done in %f seconds.\n", sec(int_mul));
}
