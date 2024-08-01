#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <thread>

#include <random>

#include <unistd.h>

#define print_matrix(x) std::cout << x.format() << std::endl
#define print_shape(x) std::cout << x.shape() << std::endl

class Matrix {
public:
  Matrix(size_t, size_t);

  void setValue(size_t row, size_t column, float value);
  Matrix dot(const Matrix&) const;
  Matrix mul(const Matrix&, unsigned int) const;
  Matrix mul(float) const;
  Matrix parallel_mul(const Matrix&, size_t, unsigned int) const;
  Matrix slow_mul(const Matrix&) const;
  Matrix operator+(const Matrix&) const;
  Matrix operator-(const Matrix&) const;

  const float &operator()(size_t row, size_t col) const;
  float &operator()(size_t row, size_t col);

  // deprecated
  const float *operator[](size_t rows) const;

  Matrix transpose() const;

  void randomise(float, float);

  const std::string format() const;
  const std::string shape() const;

  size_t rows() const;
  size_t cols() const;

  const std::vector<float>& getData() const;
  std::vector<float>& getData();

  ~Matrix();

private:
  std::vector<float> data;
  //float *data;
  size_t rowCount;
  size_t colCount;
};
