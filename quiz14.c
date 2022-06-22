void add(int n, const float *x, float *o)
{
    // o[i] += x[i] を実行
    for (int i = 0; i < n; i++)
    {
        o[i] += x[i];
    }
}
void scale(int n, float x, float *o)
{
    // o[i] *= x を実行
    for (int i = 0; i < n; i++)
    {
        o[i] *= x;
    }
}
void init(int n, float x, float *o)
{
    // o[i] = x を実行
    for (int i = 0; i < n; i++)
    {
        o[i] = x;
    }
}
void rand_init(int n, float *o)
{
    // o[i] を [-1:1] の乱数で初期化
    for (int i = 0; i < n; i++)
    {
        o[i] = (float)rand() / RAND_MAX * 2 - 1;
    }
}
