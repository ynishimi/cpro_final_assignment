#include "nn.h" int main() {
// 省略
int i = 0;
save_mnist_bmp(train_x + 784*i, "train_%05d.bmp", i); return 0;
}