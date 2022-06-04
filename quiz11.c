#include "nn.h"

//行列を表示する (quiz1.c)
void print(int m, int n, const float *x)
{
    int i, j, count;
    count = 0;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%7.4f ", x[count]);
            count++;
        }
        putchar('\n');
    }
}


//式 (1) を計算する (quiz2.c)
void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{
    int i, j;
    // yはm行で、それぞれの要素について式を適用
    for (i = 0; i < m; i++)
    {
        y[i] = b[i];
        //それぞれの要素についての計算
        for (j = 0; j < n; j++)
        {
            y[i] += A[n * i + j] * x[j];
        }
    }
}

//式 (2) を計算 (quiz3.c)
void relu(int n, const float *x, float *y)
{
    int i;
    for (i = 0; i < n; i++)
    {
        y[i] = (x[i] > 0 ? x[i] : 0);
    }
}

//式(4) を計算(quiz4.c)
void softmax(int n, const float *x, float *y)
{
    int i, j, k;
    float max_x, sum;
    max_x = 0;
    sum = 0;
    float exp_x[n];

    // max_x求める
    for (i = 0; i < n; i++)
    {
        if (max_x < x[i])
        {
            max_x = x[i];
        }
    }

    //分子を計算
    for (j = 0; j < n; j++)
    {
        exp_x[j] = exp(x[j] - max_x);
        sum += exp_x[j];
    }
    for (k = 0; k < n; k++)
    {
        y[k] = exp_x[k] / sum;
    }
}

//入れた要素のうち最大の添え字を返す (quiz5.c)
int inference3(const float *A, const float *b, const float *x, float *y)
{
    int ans;
    float max_x = 0;
    fc(10, 784, x, A, b, y);
    relu(10, y, y);
    softmax(10, y, y);

    for (int i = 0; i < 10; i++)
    {
        if (max_x < y[i])
        {
            max_x = y[i];
            ans = i;
        }
    }
    return ans;
}

void backward3(const float *A, const float *b, const float *x, unsigned char t,
               float *y, float *dEdA, float *dEdb)
{
    inference3(A, b, x, y);
}
int main()
{
    //データの準備
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count,
               &test_x, &test_y, &test_count,
               &width, &height);


    float *y = malloc(sizeof(float) * 10);
    float *dEdA = malloc(sizeof(float) * 784 * 10);
    float *dEdb = malloc(sizeof(float) * 10);
    backward3(A_784x10, b_784x10, train_x + 784 * 8, train_y[8], y, dEdA, dEdb);
   //test
    print(1, 10, y);
    // print(10, 784, dEdA);
    // print(1, 10, dEdb);
    return 0;
}
