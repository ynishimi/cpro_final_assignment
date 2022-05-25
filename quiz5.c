#include "nn.h"

#define A_M 10
#define A_N 784


int inference3(const float *A, const float *b, const float *x) 
{
    int ans;
    float y[A_M];
    int i, j, k;
    float max_x, sum;
        float exp_x[A_N];

    // fc

    // yはm行で、それぞれの要素について式を適用
    for (i = 0; i < A_M; i++)
    {
        y[i] = b[i];
        //それぞれの要素についての計算
        for (j = 0; j < A_N; j++)
        {
            y[i] += A[A_N * i + j] * x[j];
        }
    }


    //ReLU
    for (i = 0; i < A_N; i++)
    {
        y[i] = (y[i] > 0 ? y[i] : 0);
    }


    //Softmax
    max_x = 0;
    sum = 0;


    //max_x求める
    for (i = 0; i < A_N; i++)
    {
        if (max_x < y[i])
        {
            max_x = y[i];
        }
    }

    //分子を計算
    for (j = 0; j < A_N; j++)
    {
        exp_x[j] = exp(y[j] - max_x);
        sum += exp_x[j];
    }
    for (k = 0; k < A_N; k++)
    {
        y[k] = exp_x[k] / sum;
    }


    //推論
    ans = 0;
    for (i = 0; i < A_N; i++)
    {
        if (ans < y[i])
        {
            ans = y[i];
        }
    }

    //推論結果を返す
    return ans;
}
// テスト
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
    int ans = inference3(A_784x10, b_784x10, train_x);
    printf("%d %d\n", ans, train_y[0]);
    return 0;
}