#include <time.h>
#include "nn.h"

#define PI 3.14159

#define NUMBER_A1_ROW 50
#define NUMBER_A1_COLUMN 784
#define NUMBER_A2_ROW 100
#define NUMBER_A2_COLUMN 50
#define NUMBER_A3_ROW 10
#define NUMBER_A3_COLUMN 100

#define EPOCH 10
#define MINIBATCH 100 //画像を何枚づつ使用するか
#define LEARN_RATE 0.1
#define N 60000 //画像の枚数

//学習したパラメータを保存
void save(const char *filename, int m, int n,
          const float *A, const float *b)
{
    FILE *fp;
    fp = fopen(filename, "wb");
    if (fp == NULL)
        printf("Cannot open a file\n");

    else
    {
        printf("Saved\n");

        fwrite(A, sizeof(float), m * n, fp);
        fwrite(b, sizeof(float), m, fp);
        fclose(fp);
    }
}

//学習したパラメータを読み込む
void load(const char *filename, int m, int n,
          float *A, float *b)
{
    FILE *fp;
    fp = fopen(filename, "rb");
    if (fp == NULL)
        printf("Cannot open a file\n");
    else
    {

        fread(A, sizeof(float), m * n, fp);
        fread(b, sizeof(float), m, fp);
        fclose(fp);
    }
}

//行列を表示する
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

// Octave形式で行列を表示
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

// fcを計算する
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

// ReLUを計算
void relu(int n, const float *x, float *y)
{
    int i;
    for (i = 0; i < n; i++)
    {
        y[i] = (x[i] > 0 ? x[i] : 0);
    }
}
// Softmaxを計算
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
    for (int i = 0; i < n; i++)
    {
        y[i] = exp_x[i] / sum;
    }
}

//推論
void inference6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, float *out_fc1, float *out_relu1, float *out_fc2, float *out_relu2, float *out_fc3, float *y)
{
    // FC1
    fc(NUMBER_A1_ROW, NUMBER_A1_COLUMN, x, A1, b1, out_fc1);
    // ReLU1
    relu(NUMBER_A1_ROW, out_fc1, out_relu1);
    // FC2
    fc(NUMBER_A2_ROW, NUMBER_A2_COLUMN, out_relu1, A2, b2, out_fc2);
    // ReLU2
    relu(NUMBER_A2_ROW, out_fc2, out_relu2);
    // FC3
    fc(NUMBER_A3_ROW, NUMBER_A3_COLUMN, out_relu2, A3, b3, out_fc3);
    softmax(10, out_fc3, y);
}

// Softmaxの逆
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
            dEdx[i] = y[i];
        }
    }
}

// ReLUの逆
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
// fcの逆
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

//誤差逆伝搬
void backward6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, float *out_fc1, float *out_relu1, float *out_fc2, float *out_relu2, float *out_fc3, unsigned char t,
               float *y, float *dEdA1, float *dEdb1, float *dEdA2, float *dEdb2, float *dEdA3, float *dEdb3)
{

    float *dEdx3 = malloc(sizeof(float) * NUMBER_A3_ROW);
    float *dEdx2 = malloc(sizeof(float) * NUMBER_A2_ROW);
    float *dEdx1 = malloc(sizeof(float) * NUMBER_A1_ROW);
    float *dEdx = malloc(sizeof(float) * NUMBER_A1_COLUMN);

    inference6(A1, b1, A2, b2, A3, b3, x, out_fc1, out_relu1, out_fc2, out_relu2, out_fc3, y);

    // softmax
    softmaxwithloss_bwd(NUMBER_A3_ROW, y, t, dEdx3);

    // fc3
    fc_bwd(NUMBER_A3_ROW, NUMBER_A3_COLUMN, out_relu2, dEdx3, A3, dEdA3, dEdb3, dEdx2);

    // relu2
    relu_bwd(NUMBER_A2_ROW, out_fc2, dEdx2, dEdx2);

    // fc2
    fc_bwd(NUMBER_A2_ROW, NUMBER_A2_COLUMN, out_relu1, dEdx2, A2, dEdA2, dEdb2, dEdx1);

    // relu1
    relu_bwd(NUMBER_A1_ROW, out_fc1, dEdx1, dEdx1);

    // fc1
    fc_bwd(NUMBER_A1_ROW, NUMBER_A1_COLUMN, x, dEdx1, A1, dEdA1, dEdb1, dEdx);

    free(dEdx);
    free(dEdx1);
    free(dEdx2);
    free(dEdx3);
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

void nrand_init(int n, float *o, float ave, float dist)

{
    // o[i] を [-1:1] のガウス分布に従う乱数で初期化
    float x, y, z;

    for (int i = 0; i < n; i++)
    {
        x = ((float)rand() + 1.0) / (RAND_MAX + 1.0);
        y = ((float)rand() + 1.0) / (RAND_MAX + 1.0);

        z = sqrt(-2 * log(x)) * cos(2 * PI * y);

        o[i] = ave + dist * z;
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

int main(int argc, char *argv[])
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

    float *y = malloc(sizeof(float) * 10);
    float *dEdA1 = malloc(sizeof(float) * (NUMBER_A1_ROW * NUMBER_A1_COLUMN));
    float *dEdb1 = malloc(sizeof(float) * NUMBER_A1_ROW);
    float *dEdA2 = malloc(sizeof(float) * (NUMBER_A2_ROW * NUMBER_A2_COLUMN));
    float *dEdb2 = malloc(sizeof(float) * NUMBER_A2_ROW);
    float *dEdA3 = malloc(sizeof(float) * (NUMBER_A3_ROW * NUMBER_A3_COLUMN));
    float *dEdb3 = malloc(sizeof(float) * NUMBER_A3_ROW);
    float *A1 = malloc(sizeof(float) * (NUMBER_A1_ROW * NUMBER_A1_COLUMN));
    float *b1 = malloc(sizeof(float) * NUMBER_A1_ROW);
    float *A2 = malloc(sizeof(float) * (NUMBER_A2_ROW * NUMBER_A2_COLUMN));
    float *b2 = malloc(sizeof(float) * NUMBER_A2_ROW);
    float *A3 = malloc(sizeof(float) * (NUMBER_A3_ROW * NUMBER_A3_COLUMN));
    float *b3 = malloc(sizeof(float) * NUMBER_A3_ROW);
    int *index = malloc(sizeof(int) * N);

    float *out_fc1 = malloc(sizeof(float) * NUMBER_A1_ROW);
    float *out_relu1 = malloc(sizeof(float) * NUMBER_A1_ROW);
    float *out_fc2 = malloc(sizeof(float) * NUMBER_A2_ROW);
    float *out_relu2 = malloc(sizeof(float) * NUMBER_A2_ROW);
    float *out_fc3 = malloc(sizeof(float) * NUMBER_A3_ROW);

    //平均勾配を定義
    float *dEdA1_ave = malloc(sizeof(float) * (NUMBER_A1_ROW * NUMBER_A1_COLUMN));
    float *dEdA2_ave = malloc(sizeof(float) * (NUMBER_A2_ROW * NUMBER_A2_COLUMN));
    float *dEdA3_ave = malloc(sizeof(float) * (NUMBER_A3_ROW * NUMBER_A3_COLUMN));
    float *dEdb1_ave = malloc(sizeof(float) * NUMBER_A1_ROW);
    float *dEdb2_ave = malloc(sizeof(float) * NUMBER_A2_ROW);
    float *dEdb3_ave = malloc(sizeof(float) * NUMBER_A3_ROW);

    // モードの切り替え

    // 0: 学習
    if (*argv[1] == '0')
    {

        // A, bの初期化
        nrand_init(NUMBER_A1_ROW * NUMBER_A1_COLUMN, A1, 0, sqrt(2.0 / NUMBER_A1_COLUMN));
        nrand_init(NUMBER_A2_ROW * NUMBER_A2_COLUMN, A2, 0, sqrt(2.0 / NUMBER_A2_COLUMN));
        nrand_init(NUMBER_A3_ROW * NUMBER_A3_COLUMN, A3, 0, sqrt(2.0 / NUMBER_A3_COLUMN));
        init(NUMBER_A1_ROW, 0, b1);
        init(NUMBER_A2_ROW, 0, b2);
        init(NUMBER_A3_ROW, 0, b3);

        //エポック回数だけ以下の処理を繰り返す
        for (int i = 0; i < EPOCH; i++)
        {
            printf("<Epoch%2d>\n", i + 1);

            // indexのならびかえ

            for (int j = 0; j < N; j++)
            {
                index[j] = j;
            }
            shuffle(N, index);
            int sum_learn = 0;
            float E_sum_learn = 0;

            //ミニバッチ学習
            for (int j = 0; j < (N / MINIBATCH); j++)
            {
                printf("\rlearning(%3d/%3d)", j, (N / MINIBATCH));
                //平均勾配の初期化
                init((NUMBER_A1_ROW * NUMBER_A1_COLUMN), 0, dEdA1_ave);
                init((NUMBER_A2_ROW * NUMBER_A2_COLUMN), 0, dEdA2_ave);
                init((NUMBER_A3_ROW * NUMBER_A3_COLUMN), 0, dEdA3_ave);
                init(NUMBER_A1_ROW, 0, dEdb1_ave);
                init(NUMBER_A2_ROW, 0, dEdb2_ave);
                init(NUMBER_A3_ROW, 0, dEdb3_ave);

                // indexから次のMINIBATCH個とりだして、MINIBATCH回勾配を計算
                for (int k = 0; k < MINIBATCH; k++)
                {

                    // シャッフルした画像の番号: index[j * MINIBATCH + k]
                    backward6(A1, b1, A2, b2, A3, b3, train_x + 784 * index[j * MINIBATCH + k], out_fc1, out_relu1, out_fc2, out_relu2, out_fc3, train_y[index[j * MINIBATCH + k]], y, dEdA1, dEdb1, dEdA2, dEdb2, dEdA3, dEdb3);

                    //学習時の正解率を表示

                    int ans = 0;
                    float max_x = 0;

                    for (int l = 0; l < 10; l++)
                    {
                        if (max_x < y[l])
                        {
                            max_x = y[l];
                            ans = l;
                        }
                    }

                    if (ans == train_y[index[j * MINIBATCH + k]])
                    {
                        sum_learn++;
                    }

                    E_sum_learn += cross_entropy_error(y, train_y[index[j * MINIBATCH + k]]);

                    //平均勾配に計算結果を加える
                    add((NUMBER_A1_ROW * NUMBER_A1_COLUMN), dEdA1, dEdA1_ave);
                    add((NUMBER_A2_ROW * NUMBER_A2_COLUMN), dEdA2, dEdA2_ave);
                    add((NUMBER_A3_ROW * NUMBER_A3_COLUMN), dEdA3, dEdA3_ave);
                    add(NUMBER_A1_ROW, dEdb1, dEdb1_ave);
                    add(NUMBER_A2_ROW, dEdb2, dEdb2_ave);
                    add(NUMBER_A3_ROW, dEdb3, dEdb3_ave);
                }
                // MINIBATCHで割って平均勾配を完成させる

                // A, bの更新(平均勾配にLEARN_RATEかけてもとの行列から引き算)
                scale((NUMBER_A1_ROW * NUMBER_A1_COLUMN), (-LEARN_RATE / MINIBATCH), dEdA1_ave);
                scale((NUMBER_A2_ROW * NUMBER_A2_COLUMN), (-LEARN_RATE / MINIBATCH), dEdA2_ave);
                scale((NUMBER_A3_ROW * NUMBER_A3_COLUMN), (-LEARN_RATE / MINIBATCH), dEdA3_ave);
                scale(NUMBER_A1_ROW, (-LEARN_RATE / MINIBATCH), dEdb1_ave);
                scale(NUMBER_A2_ROW, (-LEARN_RATE / MINIBATCH), dEdb2_ave);
                scale(NUMBER_A3_ROW, (-LEARN_RATE / MINIBATCH), dEdb3_ave);

                add((NUMBER_A1_ROW * NUMBER_A1_COLUMN), dEdA1_ave, A1);
                add((NUMBER_A2_ROW * NUMBER_A2_COLUMN), dEdA2_ave, A2);
                add((NUMBER_A3_ROW * NUMBER_A3_COLUMN), dEdA3_ave, A3);
                add(NUMBER_A1_ROW, dEdb1_ave, b1);
                add(NUMBER_A2_ROW, dEdb2_ave, b2);
                add(NUMBER_A3_ROW, dEdb3_ave, b3);
            }

            //学習時の損失関数と正解率をエポックごとに表示
            printf("\r損失関数(訓練データ)%d: %f\n", i + 1, E_sum_learn * 100.0 / N);
            printf("認識精度(訓練データ)%d: %f%%\n", i + 1, sum_learn * 100.0 / N);

            //テストデータで推論
            //正解率を表示
            int sum = 0;
            float E_sum = 0;
            for (int j = 0; j < test_count; j++)
            {
                int ans = 0;
                float max_x = 0;

                inference6(A1, b1, A2, b2, A3, b3, test_x + j * width * height, out_fc1, out_relu1, out_fc2, out_relu2, out_fc3, y);

                for (int k = 0; k < 10; k++)
                {
                    if (max_x < y[k])
                    {

                        max_x = y[k];
                        ans = k;
                    }
                }

                if (ans == test_y[j])
                {
                    sum++;
                }

                E_sum += cross_entropy_error(y, test_y[j]);
            }
            printf("損失関数(テスト)%d: %f\n", i + 1, E_sum * 100.0 / test_count);
            printf("正解率(テスト)%d: %f%%\n///\n", i + 1, sum * 100.0 / test_count);

            //損失関数が10より小さくなったとき訓練を打ち切ってパラメータを保存

            if ((E_sum * 100.0 / test_count) < 10)
            {
                printf("finished learning\n");
                break;
            }
        }

        save("fc1.dat", 50, 784, A1, b1);
        save("fc2.dat", 100, 50, A2, b2);
        save("fc3.dat", 10, 100, A3, b3);
    }

    // 1: 推論
    else if (*argv[1] == '1')
    {
        load(argv[2], 50, 784, A1, b1);
        load(argv[3], 100, 50, A2, b2);
        load(argv[4], 10, 100, A3, b3);
        float *x = load_mnist_bmp(argv[5]);

        //正解率を表示
        int ans = 0;
        float max_x = 0;

        inference6(A1, b1, A2, b2, A3, b3, x, out_fc1, out_relu1, out_fc2, out_relu2, out_fc3, y);

        for (int k = 0; k < 10; k++)
        {
            if (max_x < y[k])
            {

                max_x = y[k];
                ans = k;
            }
        }

        //認識
        printf("ans = %d\n", ans);
    }
    else
    {
        printf("invalid input\n");
    }

    free(y);
    free(dEdA1);
    free(dEdb1);
    free(dEdA2);
    free(dEdb2);
    free(dEdA3);
    free(dEdb3);

    free(A1);
    free(b1);
    free(A2);
    free(b2);
    free(A3);
    free(b3);
    free(index);

    free(out_fc1);
    free(out_relu1);
    free(out_fc2);
    free(out_relu2);
    free(out_fc3);

    free(dEdA1_ave);
    free(dEdA2_ave);
    free(dEdA3_ave);
    free(dEdb1_ave);
    free(dEdb2_ave);
    free(dEdb3_ave);
    return 0;
}
