#include <stdio.h>

#define FOURCC_ARGS(x)  (char)((x & 0xff000000) >> 24), \
		(char)((x & 0xff0000) >> 16),					\
		(char)((x & 0xff00) >> 8), (char)((x) & 0xff)


int main()
{
	printf("code: %c%c%c%c\n", FOURCC_ARGS(1934587252)); // 'sOut'
}