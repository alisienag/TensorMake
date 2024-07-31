#include "./include/matrix.h"

int main() {
  Matrix matrix(3, 3);
  matrix.setValue(0, 0, 1);
  matrix.setValue(0, 1, 2);
  matrix.setValue(1, 0, 3);
  matrix.setValue(1, 1, 4);

  Matrix matrix2(3, 3);
  matrix2.setValue(0, 2, 1);
  matrix2.setValue(1, 2, 1);
  matrix2.setValue(2, 2, 1);
  matrix2.setValue(2, 1, 1);
  matrix2.setValue(2, 0, 1);

  matrix2.randomise(-0.5f, 0.5f);

  print_matrix(matrix2);

  print_matrix(matrix.mul(matrix2));
  print_matrix(matrix.slow_mul(matrix2));

  return 69;
}
