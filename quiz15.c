#include <time.h>
#include "nn.h"

#define NUMBER_A_ROW 10
#define NUMBER_A_COLUMN 784
#define NUMBER_FC_X 784
#define NUMBER_RELU_X 10
#define NUMBER_ANS 10

#define EPOCH 10
#define MINIBATCH 100 //画像を何枚づつ使用するか
#define LEARN_RATE 0.1
#define N 60000 //画像の枚数

//行列を表示する (quiz1.c)
void print(int m, int n, const float *x)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // i行j列の要素をプリントするとき、n * i + j番目を参照
            printf("%6.4f ", x[n * i + j]);
        }
        putchar('\n');
    }
}

void print_oct(int m, int n, const float *x, const char *name)
{
    printf("%s = [ ", name);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // i行j列の要素をプリントするとき、n * i + j番目を参照
            printf("%6.4f ", x[n * i + j]);
        }
        printf(";\n");
    }
    printf("];\n");
}

//式 (1) を計算する (quiz2.cより)
void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{
    // print_oct(m, n, A, "A");
    //  y(n個)を受け取ってy(m個)を出力する場合に対応
    float *input = malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++)
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
    float max_x, sum;
    max_x = 0;
    sum = 0;
    float exp_x[n];

    // max_x求める
    for (int i = 0; i < n; i++)
    {
        if (max_x < x[i])
        {
            max_x = x[i];
        }
    }

    //分子を計算
    for (int i = 0; i < n; i++)
    {
        exp_x[i] = exp(x[i] - max_x);
        sum += exp_x[i];
    }
    for (int i = 0; i < n;i++)
    {
        y[i] = exp_x[i] / sum;
    }
}

//入れた要素のうち最大の添え字を返す (quiz5.cより、yをmain関数から取得するように改造)
void inference3(const float *A, const float *b, const float *x, float *out_fc, float *out_relu, float *y)
{

    fc(NUMBER_A_ROW, NUMBER_A_COLUMN, x, A, b, out_fc);

    relu(NUMBER_RELU_X, out_fc, out_relu);

    softmax(10, out_relu, y);

    // test
    // printf("y=");
    // print(1, 10, y);
}

//出力からSoftmax, ReLUへ(quiz8.c)

void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx)
{
    //ここでxはSoftmaxに入ってくる10個の数、yは出力（10個）
    // t は正解の時だけ1でそれ以外は0だから、出力のうち正解の時だけ1引く
    int t_int = (int)t;
    for (int i = 0; i < n; i++)
    {
        if (i == t_int)
        {
            dEdx[i] = y[i] - 1.0;
        }
        else
        {
            dEdx[i] = y[i] - 0;
        }
    }
}

// ReLUからFCへ(quiz9.c)
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx) // dEdxはyと同じ大きさの配列へのポインタ
{
    // print_oct(10, 1, x, "x");

    for (int i = 0; i < n; i++)
    {

        if (x[i] > 0)
        {
            dEdx[i] = dEdy[i];
        }
        else
        {
            dEdx[i] = 0;
        }
    }
}

//(quiz10.c)ここが壊れてる
void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A, float *dEdA, float *dEdb, float *dEdx)
{

    // Aはm*n行列, Bはm次元ベクトル, xはn次元ベクトル
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            dEdA[n * i + j] = dEdy[i] * x[j];
        }
    }

    for (int i = 0; i < m; i++)
    {
        dEdb[i] = dEdy[i];
    }

    
        //(6層のときdEdx, dEdyがおなじポインタを扱うことを防ぐため)ReLUの逆から出力されたdEdx(m個)を受け取ってinputに入れて(これをdEdyとする)、dEdx(n個)を計算
        float *input = malloc(sizeof(float) * m);
        for (int i = 0; i < m; i++)
        {
            input[i] = dEdy[i];
        }
    
    for (int i = 0; i < n; i++)
    {
        dEdx[i] = 0;
        for (int j = 0; j < m; j++)
        {
            dEdx[i] += input[j] * A[n * j + i];
        }
    }
}

void backward3(const float *A, const float *b, const float *x, unsigned char t,
               float *out_fc, float *out_relu, float *y, float *dEdA, float *dEdb)
{
    float *dEdx = malloc(sizeof(float) * NUMBER_A_COLUMN);

    inference3(A, b, x, out_fc, out_relu, y);

    // test
    // printf("t = %d\ninference3: \n", t);
    // print_oct(10, 1, y, "y");

    softmaxwithloss_bwd(NUMBER_ANS, y, t, dEdx);

    // test
    // printf("softmaxwithloss_bwd:\n");
    // print_oct(10, 1, dEdx, "dEdx");

    relu_bwd(NUMBER_ANS, out_fc, dEdx, dEdx);

    // test
    // printf("relu_bwd:\n");
    // print_oct(10, 1, dEdx, "dEdx");

    fc_bwd(NUMBER_A_ROW, NUMBER_A_COLUMN, x, dEdx, A, dEdA, dEdb, dEdx);

    // test
    // printf("fc_bwd:\n");
    // print_oct(10, 1, dEdb, "dEdb");

    free(dEdx);
}

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

void shuffle(int n, int *x)
{
    int temp;
    for (int i = 0; i < n; i++)
    {
        int j = rand() % n;

        //配列の個別の要素をスワップ
        temp = *(x + i);
        *(x + i) = *(x + j);
        *(x + j) = temp;
    }
}

float cross_entropy_error(const float *y, int t)
{
    return -log(y[t] + 1e-7);
}

int main()
{
    srand(time(NULL));

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
    float *out_fc = malloc(sizeof(float) * NUMBER_RELU_X);
    float *out_relu = malloc(sizeof(float) * 10);
    float *y = malloc(sizeof(float) * 10);
    float *dEdA = malloc(sizeof(float) * (NUMBER_A_ROW * NUMBER_A_COLUMN));
    float *dEdb = malloc(sizeof(float) * 10);
    float *A = malloc(sizeof(float) * (NUMBER_A_ROW * NUMBER_A_COLUMN));
    float *b = malloc(sizeof(float) * 10);
    int *index = malloc(sizeof(int) * N);

    //平均勾配を定義
    float *dEdA_ave = malloc(sizeof(float) * (NUMBER_A_ROW * NUMBER_A_COLUMN));
    float *dEdb_ave = malloc(sizeof(float) * 10);

    // A, bの初期化
    rand_init(NUMBER_A_ROW * NUMBER_A_COLUMN, A);
    rand_init(10, b);

    //エポック回数だけ以下の処理を繰り返す
    for (int i = 0; i < EPOCH; i++)
    {
        // indexのならびかえ

        for (int j = 0; j < N; j++)
        {
            index[j] = j;
        }
        shuffle(N, index);

        //ミニバッチ学習
        for (int j = 0; j < (N / MINIBATCH); j++)
        {

            //平均勾配の初期化
            init((NUMBER_A_ROW * NUMBER_A_COLUMN), 0, dEdA_ave);
            init(10, 0, dEdb_ave);

            // indexから次のMINIBATCH個とりだして、MINIBATCH回勾配を計算
            for (int k = 0; k < MINIBATCH; k++)
            {

                // シャッフルした画像の番号: index[j * MINIBATCH + k]
                backward3(A, b, train_x + 784 * index[j * MINIBATCH + k], train_y[index[j * MINIBATCH + k]], out_fc, out_relu,  y, dEdA, dEdb);

                // test
                // printf("index[%d]\n", index[j * MINIBATCH + k]);
                // print(1, 10, dEdb);

                //平均勾配に計算結果を加える
                add((NUMBER_A_ROW * NUMBER_A_COLUMN), dEdA, dEdA_ave);
                add(10, dEdb, dEdb_ave);

                // test
                // print_oct(10, 1, dEdb_ave, "dEdb_ave");
            }

            // MINIBATCHで割って平均勾配を完成させる

            // test
            // printf("MINI_%d\n", j);

            scale((NUMBER_A_ROW * NUMBER_A_COLUMN), (1.0 / MINIBATCH), dEdA_ave);
            scale(10, (1.0 / MINIBATCH), dEdb_ave);
            // print_oct(10, 1, dEdb_ave, "dEdb_ave");

            // A, bの更新(平均勾配にLEARN_RATEかけてもとの行列から引き算)
            scale((NUMBER_A_ROW * NUMBER_A_COLUMN), -LEARN_RATE, dEdA_ave);
            scale(10, -LEARN_RATE, dEdb);

            // test
            // print_oct(10, 1, b, "b");

            add((NUMBER_A_ROW * NUMBER_A_COLUMN), dEdA_ave, A);
            add(10, dEdb_ave, b);

            // test

            // print_oct(10, 1, b, "b_new");

            // test
            // printf("batch(%d)\n", j);
            // print(1, 10, dEdb_ave);
        }

        
        //テストデータで推論

        //正解率を表示
        int sum = 0;
        float E_sum = 0;
        for (int j = 0; j < test_count; j++)
        {
            int ans = 0;
            float max_x = 0;


            inference3(A, b, test_x + j * width * height,out_fc, out_relu, y);

            for (int k = 0; k < 10; k++)
            {
                if (max_x < y[k])
                {

                    max_x = y[k];
                    ans = k;
                }
            }
            // printf("inf(%d):\n", j);
            // print(1, 10, y);
            // printf("%d, %d\n///\n", ans, test_y[j]);
            if (ans == test_y[j])
            {
                sum++;
            }

            //損失関数
            // printf("%f\n", cross_entropy_error(y, test_y[j]));

            E_sum += cross_entropy_error(y, test_y[j]);
        }
        printf("損失関数%d: %f\n", i + 1, E_sum * 100.0 / test_count);
        printf("正解率%d: %f%%\n", i + 1, sum * 100.0 / test_count);
    }







    return 0;
}
