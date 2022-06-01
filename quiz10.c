void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A, float *dEdA, float *dEdb, float *dEdx);
{
    // Aはm*n行列, Bはm次元ベクトル, xはn次元ベクトル
    for (int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
                    dEdA[] = dEdy[i] * 
        }
    }
//書きかけ
            for (int i = 0; i < m; i++){
                dEdb[i] = dEdy[i]}
}