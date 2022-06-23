#include "nn.h"
// time.hがインクルードできない？？
#define NUMBER_A_ROW 10
#define NUMBER_A_COLUMN 784
#define NUMBER_FC_X 784
#define NUMBER_RELU_X 10
#define NUMBER_ANS 10

#define EPOCH 10
#define MINIBATCH 100 //画像を何枚づつ使用するか
#define LEARN_RATE 0.1
#define N 60000 //画像の枚p数

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

//式 (1) を計算する (quiz2.cより)
void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{

    // yはm行で、それぞれの要素について式を適用
    for (int i = 0; i < m; i++)
    {

        y[i] = b[i];
        //それぞれの要素についての計算
        for (int j = 0; j < n; j++)
        {
            y[i] += A[n * i + j] * x[j];
        }
    }
}

//式 (2) を計算 (quiz3.c)
void relu(int n, const float *x, float *y, float *relu_x)
{
    int i;
    for (i = 0; i < n; i++)
    {
        relu_x[i] = x[i];
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

//入れた要素のうち最大の添え字を返す (quiz5.cより、yをmain関数から取得するように改造)
void inference3(const float *A, const float *b, const float *x, float *y, float *relu_x)
{
    float max_x = 0;
    fc(NUMBER_A_ROW, NUMBER_A_COLUMN, x, A, b, y);

    relu(NUMBER_RELU_X, y, y, relu_x);

    softmax(10, y, y);

    for (int i = 0; i < 10; i++)
    {
        if (max_x < y[i])
        {
            max_x = y[i];
        }
    }
}

//出力からSoftmax, ReLUへ(quiz8.c)

void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx)
{
    //ここでxはSoftmaxに入ってくる10個の数、yは出力（10個）
    // t は正解の時だけ1でそれ以外は0だから、出力のうち正解の時だけ1引く
    for (int i = 0; i < n; i++)
    {
        if (i == t)
        {
            dEdx[i] = y[i] - 1;
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

//(quiz10.c)
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

    for (int i = 0; i < n; i++)
    {
        dEdx[i] = 0;
        for (int j = 0; j < m; j++)
        {
            dEdx[i] += dEdy[j] * A[n * j + i];
        }
    }
}

void backward3(const float *A, const float *b, const float *x, unsigned char t,
               float *y, float *dEdA, float *dEdb)
{
    float *relu_x = malloc(sizeof(float) * NUMBER_RELU_X);
    float *dEdx = malloc(sizeof(float) * 10);
    inference3(A, b, x, y, relu_x);

    softmaxwithloss_bwd(NUMBER_ANS, y, t, dEdx);
    relu_bwd(NUMBER_ANS, relu_x, dEdx, dEdx);

    fc_bwd(NUMBER_A_ROW, NUMBER_A_COLUMN, x, dEdx, A, dEdA, dEdb, dEdx);
    free(relu_x);

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
    return -log(y[t] + exp(1) - 7);
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
    float *dEdA = malloc(sizeof(float) * (NUMBER_A_ROW * NUMBER_A_COLUMN));
    float *dEdb = malloc(sizeof(float) * 10);
    float *A = malloc(sizeof(float) * (NUMBER_A_ROW * NUMBER_A_COLUMN));
    float *b = malloc(sizeof(float) * 10);

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
        srand(time(NULL));
        int *index = malloc(sizeof(int) * train_count);
        for (int i = 0; i < N; i++)
        {
            index[i] = i;
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
                backward3(A, b, train_x + 784 * index[j * MINIBATCH + k], train_y[index[j * MINIBATCH + k]], y, dEdA, dEdb);

                //平均勾配に計算結果を加える
                add((NUMBER_A_ROW * NUMBER_A_COLUMN), dEdA, dEdA_ave);
                add(10, dEdb, dEdb_ave);
            }

            // MINIBATCHで割って平均勾配を完成させる
            scale((NUMBER_A_ROW * NUMBER_A_COLUMN), (-1 / MINIBATCH), dEdA_ave);
            scale(10, (-1 / MINIBATCH), dEdb_ave);

            // A, bの更新(平均勾配にLEARN_RATEかけてもとの行列から引き算)
            scale((NUMBER_A_ROW * NUMBER_A_COLUMN), LEARN_RATE, dEdA_ave);
            scale(10, LEARN_RATE, dEdb);

            add((NUMBER_A_ROW * NUMBER_A_COLUMN), dEdA_ave, A);
            add(10, dEdb_ave, b);
        }
        //テストデータで推論

        //損失関数を表示

        //正解率を表示
        int sum = 0;
        for (int j = 0; j < test_count; j++)
        {
            int ans = 0;
            int max_x = -1;
            inference3(A, b, test_x + j * width * height, y, relu_x); //なんとかする
            for (int k = 0; k < 10; k++)
            {
                if (max_x < y[k])
                {
                    max_x = y[k];
                    ans = k;
                }
            }
            if (ans == test_y[i])
                {
                    sum++;
                }
        }
        printf("正解率: %f%%\n", sum * 100.0 / test_count);
        return 0;
    }

    return 0;
}
