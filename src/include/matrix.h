#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <random>

#define print_matrix(x) std::cout << x.format() << std::endl
#define print_shape(x) std::cout << x.shape() << std::endl

class Matrix {
 public:
  Matrix(size_t, size_t);

  void setValue(size_t row, size_t column, float value);
  Matrix dot(const Matrix&) const;
  Matrix mul(const Matrix&) const;
  Matrix slow_mul(const Matrix&) const;
  Matrix operator+(const Matrix&) const;
  Matrix operator-(const Matrix&) const;
  float *operator[](size_t rows) const;

  void randomise(float, float);

  const std::string format() const;
  const std::string shape() const;

  size_t rows() const;
  size_t cols() const;

  float *getData() const;

  ~Matrix();

 private:
  float *data;
  size_t rowCount;
  size_t colCount;
};
