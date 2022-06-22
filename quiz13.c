#include "nn.h"

float cross_entropy_error(const float *y, int t)
{
    return -log(y[t] + exp(1) - 7);
}
