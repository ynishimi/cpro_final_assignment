#include <time.h>
#include "nn.h"

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

void save(const char *filename, int m, int n,
          const float *A, const float *b)
{
    FILE *fp;
    fp = fopen(filename, "wb");
    if (fp == NULL)
        printf("Cannot open a file\n");

    else
    {
        printf("Successfully saved\n");

        fwrite(A, sizeof(float), m * n, fp);
        fwrite(b, sizeof(float), m, fp);
        fclose(fp);
    }
}

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
    for (int i = 0; i < n; i++)
    {
        y[i] = exp_x[i] / sum;
    }
}

//推論 (quiz5.cより)
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

void backward6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, float *out_fc1, float *out_relu1, float *out_fc2, float *out_relu2, float *out_fc3, unsigned char t,
               float *y, float *dEdA1, float *dEdb1, float *dEdA2, float *dEdb2, float *dEdA3, float *dEdb3)
{

    // float *dEdx = malloc(sizeof(float) * NUMBER_A1_COLUMN);

    float *dEdx3 = malloc(sizeof(float) * NUMBER_A3_ROW);
    float *dEdx2 = malloc(sizeof(float) * NUMBER_A2_ROW);
    float *dEdx1 = malloc(sizeof(float) * NUMBER_A1_ROW);
    float *dEdx = malloc(sizeof(float) * NUMBER_A1_COLUMN);

    inference6(A1, b1, A2, b2, A3, b3, x, out_fc1, out_relu1, out_fc2, out_relu2, out_fc3, y);

    // test
    // printf("t = %d\ninference3: \n", t);
    // print_oct(10, 1, y, "y");

    // softmax
    softmaxwithloss_bwd(NUMBER_A3_ROW, y, t, dEdx3);

    // fc3
    fc_bwd(NUMBER_A3_ROW, NUMBER_A3_COLUMN, out_relu2, dEdx3, A3, dEdA3, dEdb3, dEdx2);

    // test
    // printf("softmaxwithloss_bwd:\n");
    // print_oct(10, 1, dEdx, "dEdx");

    // relu2
    relu_bwd(NUMBER_A2_ROW, out_fc2, dEdx2, dEdx2);

    // test
    // printf("relu_bwd:\n");
    // print_oct(10, 1, dEdx, "dEdx");

    // fc2
    fc_bwd(NUMBER_A2_ROW, NUMBER_A2_COLUMN, out_relu1, dEdx2, A2, dEdA2, dEdb2, dEdx1);

    // test
    // printf("fc_bwd:\n");
    // print_oct(10, 1, dEdb, "dEdb");

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

    float *y = malloc(sizeof(float) * 10);
    /*
    float *dEdA1 = malloc(sizeof(float) * (NUMBER_A1_ROW * NUMBER_A1_COLUMN));
    float *dEdb1 = malloc(sizeof(float) * NUMBER_A1_ROW);
    float *dEdA2 = malloc(sizeof(float) * (NUMBER_A2_ROW * NUMBER_A2_COLUMN));
    float *dEdb2 = malloc(sizeof(float) * NUMBER_A2_ROW);
    float *dEdA3 = malloc(sizeof(float) * (NUMBER_A3_ROW * NUMBER_A3_COLUMN));
    float *dEdb3 = malloc(sizeof(float) * NUMBER_A3_ROW);

    */
    float *A1 = malloc(sizeof(float) * (NUMBER_A1_ROW * NUMBER_A1_COLUMN));
    float *b1 = malloc(sizeof(float) * NUMBER_A1_ROW);
    float *A2 = malloc(sizeof(float) * (NUMBER_A2_ROW * NUMBER_A2_COLUMN));
    float *b2 = malloc(sizeof(float) * NUMBER_A2_ROW);
    float *A3 = malloc(sizeof(float) * (NUMBER_A3_ROW * NUMBER_A3_COLUMN));
    float *b3 = malloc(sizeof(float) * NUMBER_A3_ROW);
    //int *index = malloc(sizeof(int) * N);

    float *out_fc1 = malloc(sizeof(float) * NUMBER_A1_ROW);
    float *out_relu1 = malloc(sizeof(float) * NUMBER_A1_ROW);
    float *out_fc2 = malloc(sizeof(float) * NUMBER_A2_ROW);
    float *out_relu2 = malloc(sizeof(float) * NUMBER_A2_ROW);
    float *out_fc3 = malloc(sizeof(float) * NUMBER_A3_ROW);

    //テストデータで推論
    load("fc1.dat", 50, 784, A1, b1);
    load("fc2.dat", 100, 50, A2, b2);
    load("fc3.dat", 10, 100, A3, b3);

    //正解率を表示
    int sum = 0;
    float E_sum = 0;
    for (int j = 0; j < test_count; j++)
    {
        printf("\rinference(%3d/%3d)", j, test_count);
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
    printf("\r損失関数(テスト): %f\n", E_sum * 100.0 / test_count);
    printf("正解率(テスト): %f%%\n///\n", sum * 100.0 / test_count);
    //}
    return 0;
}
