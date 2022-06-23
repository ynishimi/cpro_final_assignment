#include "nn.h"

#define EPOCH 10
#define MINIBATCH 100 //画像を何枚づつ使用するか
#define LEARN_RATE 0.1 
#define N 60000 //画像の枚数

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

    //A, bの初期化

    //エポック回数だけ以下の処理を繰り返す
    for (int i = 0; i < EPOCH; i++)
    {
        //ミニバッチ学習
        for (int j = 0; j < (N / MINIBATCH); j++)
        {
            //平均勾配の初期化
            
        }
    }

        return 0;
}
