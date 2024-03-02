#include <stdio.h>

float* get_page(void);

int main(int argc, char* argv[]) {
    printf("Hello World\n");
    float* zero_vector = get_page();
    printf("zero vector=[%.4f, %.4f]\n", zero_vector[0], zero_vector[1]);
    return 0;
}
