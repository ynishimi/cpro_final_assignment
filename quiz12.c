#include "nn.h"
//time.hをインクルードできない
#include <time.h>

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

    srand(time(NULL));
    int *index = malloc(sizeof(int) * train_count);
    for (int i = 0; i < train_count; i++)
    {
        index[i] = i;
    }
    shuffle(train_count, index);
    return 0;
}
