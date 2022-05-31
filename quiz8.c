#include "nn.h"

void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx)
{

    //t は正解の時だけ1でそれ以外は0だから、出力のうち正解の時だけ1引く
    for (int i = 0; i < 10; i++)
    {
        if (i == (t-1))
        {
            dEdx[i] = y[i] - 1;
        }
        else
        {
            dEdx[i] = y[i] - 0;
        }
    }
}

int main()
{

    return 0;
}
