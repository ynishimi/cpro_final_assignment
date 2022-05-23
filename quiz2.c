#include "nn.h"

//式 (1) を計算する (quiz2.c)
void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{
    int i, j;
    //yはm行で、それぞれの要素について式を適用
    for (i = 0; i < m; i++)
    {
        y[i] = b[i];
        //それぞれの要素についての計算
        for (j = 0; j < n; j++)
        {
            y[i] += A[n * i + j];
        }
    }
}

//行列を表示する (quiz1.c)
void print(int m, int n, const float * x)
{
    int i, j, count;
    count = 0;
    for (i = 0; i < m; i++)
    {
        for(j = 0; j < n; j++)
        {
            printf("%7.4f ", x[count]);
            count++;
        }
        putchar('\n');
    }
}

// テスト
int main()
{
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
    // これ以降，３層 NN の係数 A_784x10 および b_784x10 と，
    // 訓練データ train_x + 784*i (i=0,...,train_count-1), train_y[0]～train_y[train_count-1],
    // テストデータ test_x + 784*i (i=0,...,test_count-1), test_y[0]～test_y[test_count-1],
    // を使用することができる．
    float y[10];
    fc(10, 784, train_x, A_784x10, b_784x10, y);
    print(1, 10, y);
    return 0;
}
