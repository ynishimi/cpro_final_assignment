void relu_bwd(int n, const float * x, const float * dEdy, float * dEdx) //dEdxはyと同じ大きさの配列へのポインタ
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