void backward3(const float *A, const float *b, const float *x, unsigned char t,
               float *y, float *dEdA, float *dEdb)
{
}
int main()
{
    // 省略
    float *y = malloc(sizeof(float) * 10);
    float *dEdA = malloc(sizeof(float) * 784 * 10);
    float *dEdb = malloc(sizeof(float) * 10);
    backward3(A_784x10, b_784x10, train_x + 784 * 8, train_y[8], y, dEdA, dEdb);
    print(10, 784, dEdA);
    print(1, 10, dEdb);
    return 0;
}
