#include "nn.h"

#define NUMBER_A1_ROW 50
#define NUMBER_A1_COLUMN 784
#define NUMBER_A2_ROW 100
#define NUMBER_A2_COLUMN 50
#define NUMBER_A3_ROW 10
#define NUMBER_A3_COLUMN 100

#define NUMBER_b1 50
#define NUMBER_b2 100
#define NUMBER_b1 10

#define NUMBER_ANS 10

/*
//行列を表示する (quiz1.c)
void print(int m, int n, const float *x)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            //i行j列の要素をプリントするとき、n * i + j番目を参照
            printf("%6.4f ", x[n * i + j]);
        }
        putchar('\n');
    }
}
*/

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

//入れた要素のうち最大の添え字を返す (quiz5.cより)
int inference6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, float *y)
{
    int ans;
    float max_x = 0;
   
    //FC1
    fc(NUMBER_A1_ROW, NUMBER_A1_COLUMN, x, A1, b1, y);
    //ReLU1
    relu(NUMBER_A1_ROW, y, y);
    //FC2
    fc(NUMBER_A2_ROW, NUMBER_A2_COLUMN, y, A2, b2, y);
    //ReLU2
    relu(NUMBER_A2_ROW, y, y);
    //FC3
    fc(NUMBER_A3_ROW, NUMBER_A3_COLUMN, y, A3, b3, y);
    softmax(10, y, y);

    for (int i = 0; i < 10; i++)
    {
        if (max_x < y[i])
        {
            max_x = y[i];
        }
    }

    return ans;
}


/*
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

    print(1, 10, y);
    softmaxwithloss_bwd(NUMBER_ANS, y, t, dEdx);
    relu_bwd(NUMBER_ANS, relu_x, dEdx, dEdx);

    fc_bwd(NUMBER_A_ROW, NUMBER_A_COLUMN, x, dEdx, A, dEdA, dEdb, dEdx);
    free(relu_x);

    free(dEdx);
}

*/

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
        if (inference6(A1_784_50_100_10, b1_784_50_100_10, A2_784_50_100_10, b2_784_50_100_10, A3_784_50_100_10, b3_784_50_100_10, test_x + i * width * height) == test_y[i])
        {
            sum++;
        }
    }
    printf("%f%%\n", sum * 100.0 / test_count);
    
/*
    float *y = malloc(sizeof(float) * 10);
    float *dEdA = malloc(sizeof(float) * (NUMBER_A_ROW * NUMBER_A_COLUMN));
    float *dEdb = malloc(sizeof(float) * 10);
    //backward3(A_784x10, b_784x10, train_x + 784 * 8, train_y[8], y, dEdA, dEdb);
    free(y);

     print(10, 784, dEdA);
     print(1, 10, dEdb);
    free(dEdA);
    free(dEdb);

   */
    return 0;
}
