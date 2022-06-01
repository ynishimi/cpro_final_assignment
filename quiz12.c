void shuffle(int n, int * x) {
}
int main() {
// 省略
int * index = malloc(sizeof(int)*train_count);
for(i=0 ; i<train_count ; i++) {
index[i] = i;
}
shuffle(train_count, index);
return 0;
}