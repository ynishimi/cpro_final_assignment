#include "nn.h"

//行列を表示する (quiz1.c)
void print(int m, int n, const float x[])
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

int main()
{

    print(1, 10, b_784x10);
  return 0;
}
