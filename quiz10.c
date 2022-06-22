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
        for(int j= 0; j < m; j++)
        {
            dEdx[i] += dEdy[j] * A[n * j + i];
        }
    }
}