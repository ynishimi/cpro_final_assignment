#include "nn.h"

/*(行列の表示は無効にしている)

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

*/

//式 (1) を計算する (quiz2.c)

/*
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
*/
void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{

        //(6層のときx,yがおなじポインタを扱うことを防ぐため)ReLUが出力したy(n個)を受け取って(これをxとする)、y(m個)を出力する場合に対応
    float *input = malloc(sizeof(float) * n);
    for(int i= 0; i < n; i++)
    {
        input[i] = x[i];
    }

    // yはm行で、それぞれの要素について式を適用
    for (int i = 0; i < m; i++)
    {

        y[i] = b[i];
        //それぞれの要素についての計算
        for (int j = 0; j < n; j++)
        {
            y[i] += A[n * i + j] * input[j];
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
int inference3(const float *A, const float *b, const float *x)
{
    int ans;
    float max_x = 0;
    float *y = malloc(sizeof(float) * 10);
    fc(10, 784, x, A_784x10, b_784x10, y);
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


    //正解率(quiz7.c)
    int sum = 0;
    for (int i = 0; i < test_count; i++)
    {
        if (inference3(A_784x10, b_784x10, test_x + i * width * height) == test_y[i])
        {
            sum++;
        }
    }
    printf("%f%%\n", sum * 100.0 / test_count);
    return 0;
}