int inference3(const float * A, const float * b, const float * x) { }
// テスト
int main() {
// 省略
float * y = malloc(sizeof(float)*10);
int ans = inference3(A_784x10, b_784x10, train_x); printf("%d %d\n", ans, train_y[0]);
return 0;
}