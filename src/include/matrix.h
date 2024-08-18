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
  Matrix();
  Matrix(size_t, size_t);
  explicit Matrix(std::vector<std::vector<uint8_t>>, bool);
  explicit Matrix(std::vector<float>);
  explicit Matrix(std::vector<uint8_t>);

  void setValue(size_t row, size_t column, float value);
  Matrix dot(const Matrix&) const;
  Matrix mul(const Matrix&, unsigned int) const;
  Matrix mul(float) const;
  Matrix parallel_mul(const Matrix&, size_t, unsigned int) const;
  Matrix slow_mul(const Matrix&) const;
  Matrix operator+(const Matrix&) const;
  Matrix operator-(const Matrix&) const;
  Matrix operator/(const float&) const;

  Matrix trim(size_t, size_t) const;
    float sum() const;
  Matrix sumCols() const;

  const Matrix one_hot_encode() const;
  const Matrix argmax() const;

  const float &operator()(size_t row, size_t col) const;
  const bool operator==(const Matrix& other) const;
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
  size_t rowCount;
  size_t colCount;
};
