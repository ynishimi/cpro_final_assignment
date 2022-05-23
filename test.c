#include "nn.h"

void print(int m, int n, const float * x)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for(j = 0; j < m; j++)
        {
            printf("%f\n, &x[0]");
        }
    }
}

int main()
{
    print(1, 10, b_784x10);
    return 0;
}