#include "nn.h" int main() {
// 省略
int sum = 0;
for(i=0 ; i<test_count ; i++) {
if(inference3(A_784x10, b_784x10, test_x + i*width*height) == test_y[i]) { sum++;
} }
printf("%f%%\n", sum * 100.0 / test_count);
return 0; }