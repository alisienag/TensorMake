#include "./include/matrix.h"

#include <time.h>

int main() {
  Matrix matrix(10000, 784);
  matrix.setValue(0, 0, 1);
  matrix.setValue(0, 1, 2);
  matrix.setValue(1, 0, 3);
  matrix.setValue(1, 1, 4);

  Matrix matrix2(784, 10);
  matrix2.setValue(0, 2, 1);
  matrix2.setValue(1, 2, 1);
  matrix2.setValue(2, 2, 1);
  matrix2.setValue(2, 1, 1);
  matrix2.setValue(2, 0, 1);
  
  matrix.randomise(-0.5f, 0.5f);
  matrix2.randomise(-0.5f, 0.5f);

  clock_t start = clock();
  matrix.mul(matrix2);
  clock_t end = clock();

  std::cout << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

  start = clock();
  matrix.slow_mul(matrix2);
  end = clock();

  std::cout << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

  return 69;
}
