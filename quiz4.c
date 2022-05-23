#include "nn.h"

//式(4) を計算(quiz4.c)
void softmax(int n, const float *x, float *y)
{
    int i, j, k, max_x, sum;
    max_x = 0;
    sum = 0;
    float exp_x[n];

    //max_x求める
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

//式 (2) を計算 (quiz3.c)
void relu(int n, const float *x, float *y)
{
    int i;
    for (i = 0; i < n; i++)
    {
        y[i] = (x[i] > 0 ? x[i] : 0);
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

    //テスト
    float *y = malloc(sizeof(float) * 10);

    fc(10, 784, train_x, A_784x10, b_784x10, y);
    relu(10, y, y);
    softmax(10, y, y);
    print(1, 10, y);

    return 0;
}
