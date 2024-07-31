#include "./include/matrix.h"

#include <time.h>

int main() {
  Matrix matrix(60000, 1000);
  Matrix matrix2(1000, 1000); 
  matrix.randomise(-0.5f, 0.5f);
  matrix2.randomise(-0.5f, 0.5f);
  
  std::cout << "Starting tiled multiplication!" << std::endl;
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
