#include "nn.h"

//行列を表示する
void print(int m, int n, const float * x)
{
    int i, j, count;
    count = 0;
    for (i = 0; i < m; i++)
    {
        for(j = 0; j < n; j++)
        {
            printf("%f ", x[count]);
            count++;
        }
        putchar('\n');
    }
}

int main()
{

    print(2, 5, b_784x10);
  return 0;
}
