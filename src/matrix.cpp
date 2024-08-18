#include "./include/matrix.h"
#include <locale>

Matrix::Matrix() {
  this->rowCount = 1;
  this->colCount = 1;
  this->data.push_back(0.f);
}

Matrix::Matrix(size_t rows, size_t columns) {
  this->rowCount = rows;
  this->colCount = columns;
  // this->data = new float[rows * columns];
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      this->data.push_back(0.f);
    }
  }
}

Matrix::Matrix(std::vector<std::vector<uint8_t>> data, bool normalise) {
  this->rowCount = data.size();
  this->colCount = data[0].size();

  float divisor = 1.f;

  if (normalise) {
    divisor = 255.f;
  }

  for (int i = 0; i < rowCount; i++) {
    for (int j = 0; j < colCount; j++) {
      this->data.push_back(static_cast<float>(static_cast<float>(data[i][j]) / divisor));
    }
  }
}

Matrix::Matrix(std::vector<uint8_t> data) {
  this->rowCount = 1;
  this->colCount = data.size();
  // this->data = new float[rows * columns];
  for (size_t i = 0; i < this->colCount; i++) {
    this->data.push_back(static_cast<float>(static_cast<float>(data.at(i))));
  }
}

Matrix Matrix::dot(const Matrix &other) const {
  if (this->rows() != other.rows() || this->cols() != other.cols()) {
    std::string error_message = "Matrix 'dot': ";
    error_message += this->shape() + " != " + other.shape() + "!\n";
    throw std::invalid_argument(error_message);
  }
  Matrix result(this->rowCount, this->colCount);
  for (int i = 0; i < this->rowCount * this->colCount; i++) {
    result.getData()[i] = this->data[i] * other.getData()[i];
  }
  return result;
}

Matrix Matrix::mul(const Matrix &other, unsigned int block_size) const {
  if (this->colCount != other.rows()) {
    std::string error_message = "Matrix multiplication: " + this->shape() +
                                " != " + other.shape() + "!";
    throw std::invalid_argument(error_message);
  }
  Matrix result(this->rowCount, other.cols());
  size_t n = this->rowCount;
  size_t m = this->colCount;
  size_t p = other.cols();

  for (size_t ii = 0; ii < n; ii += block_size) {
    for (size_t jj = 0; jj < p; jj += block_size) {
      for (size_t kk = 0; kk < m; kk += block_size) {
        for (size_t i = ii; i < std::min(ii + block_size, n); i++) {
          for (int j = jj; j < std::min(jj + block_size, p); j++) {
            float sum = 0.f;
            for (int k = kk; k < std::min(kk + block_size, m); k++) {
              sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) += sum;
          }
        }
      }
    }
  }

  return result;
}

Matrix Matrix::mul(float x) const {
  Matrix result(this->rowCount, this->colCount);
  for (int i = 0; i < this->rowCount * this->colCount; i++) {
    result.getData()[i] = this->data[i] * x;
  }

  return result;
}

Matrix Matrix::parallel_mul(const Matrix &other, size_t thread_count,
                            unsigned int block_size) const {
  Matrix result(this->rowCount, other.cols());

  std::vector<std::thread> threads;

  auto f = [&](unsigned int start, unsigned int end) {
    size_t n = end;
    size_t m = this->colCount;
    size_t p = other.cols();

    for (size_t ii = start; ii < n; ii += block_size) {
      for (size_t jj = 0; jj < p; jj += block_size) {
        for (size_t kk = 0; kk < m; kk += block_size) {
          for (size_t i = ii; i < std::min(ii + block_size, n); i++) {
            for (int j = jj; j < std::min(jj + block_size, p); j++) {
              float sum = 0.f;
              for (int k = kk; k < std::min(kk + block_size, m); k++) {
                sum += (*this)(i, k) * other(k, j);
              }
              result(i, j) += sum;
            }
          }
        }
      }
    }
  };
  if (this->rowCount % 2 == 0) {
    thread_count = 2;
    if (this->rowCount % 4 == 0) {
      thread_count = 4;
      if (this->rowCount % 8 == 0) {
        thread_count = 8;
        if (this->rowCount % 16 == 0) {
          thread_count = 16;
          if (this->rowCount % 32 == 0) {
            thread_count = 32;
          }
          if (this->rowCount % 64 == 0) {
            thread_count = 64;
          }
        }
      }
    }
  } else if (this->rowCount % 3 == 0) {
    thread_count = 3;
    if (this->rowCount % 6 == 0) {
      thread_count = 6;
    }
  } else {
    thread_count = 1;
  }

  unsigned int factor = this->rowCount / thread_count;

  unsigned int prev_i = 0;

  for (unsigned int i = 0; i < thread_count; i++) {
    threads.push_back(std::thread(f, prev_i, (i * factor) + factor));

    prev_i = i;
  }

  for (int i = 0; i < thread_count; i++) {
    if (threads[i].joinable())
      threads[i].join();
  }

  return result;
}

Matrix Matrix::slow_mul(const Matrix &other) const {
  if (this->colCount != other.rows()) {
    std::string error_message = "Matrix multiplication: " + this->shape() +
                                " != " + other.shape() + "!";
    throw std::invalid_argument(error_message);
  }
  Matrix result(this->rowCount, other.cols());
  int n = this->rowCount;
  int m = this->colCount;
  int p = other.cols();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      float sum = 0.f;
      for (int k = 0; k < m; k++) {
        sum += (*this)(i, j) * other(k, j);
      }
      result(i, j) = sum;
    }
  }

  return result;
}

const float &Matrix::operator()(size_t row, size_t col) const {
  if (row > this->rowCount || col > this->colCount) {
    std::stringstream ss;
    ss << "Matrix indexing: " << row << " _ " << col << std::endl;
    throw std::invalid_argument(ss.str());
  }
  return this->data[row * this->colCount + col];
}

float &Matrix::operator()(size_t row, size_t col) {
  if (row > this->rowCount || col > this->colCount) {
    std::stringstream ss;
    ss << "Matrix indexing: " << row << " _ " << col << std::endl;
    throw std::invalid_argument(ss.str());
  }
  return this->data[row * this->colCount + col];
}

const float *Matrix::operator[](size_t rows) const { // DEPRECATED
  if (rows > this->rowCount) {
    std::stringstream ss;
    ss << "Matrix indexing: " << rows << std::endl;
    throw std::invalid_argument(ss.str());
  }
  return &this->data[rows];
}

Matrix Matrix::operator+(const Matrix &other) const {
  Matrix result(this->rowCount, this->colCount);
  if (other.rows() != this->rows() || other.cols() != this->cols()) {
    if (other.cols() == this->colCount && other.rows() == 1) {
      for (int i = 0; i < this->rowCount; i++) {
        for (int j = 0; j < this->colCount; j++) {
          result(i, j) = (*this)(i, j) + other(0, j);
        }
      }
      return result;
    }
    std::string error_message =
        "Matrix addition: " + this->shape() + " != " + other.shape() + "\n";
    throw std::invalid_argument(error_message);
  } else {
    for (int i = 0; i < this->rowCount * this->colCount; i++) {
      result.getData()[i] = this->data[i] + other.getData()[i];
    }
  }

  return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
  Matrix result(this->rowCount, this->colCount);
  if (other.rows() != this->rows() || other.cols() != this->cols()) {
    std::string error_message =
        "Matix subtraction: " + this->shape() + " != " + other.shape() + "!\n";
    throw std::invalid_argument(error_message);
  } else {
    for (int i = 0; i < this->rowCount * this->colCount; i++) {
      result.getData()[i] = this->data[i] - other.getData()[i];
    }
  }

  return result;
}

Matrix Matrix::operator/(const float &divisor) const {
  Matrix result(this->rowCount, this->colCount);
  for (int i = 0; i < this->rowCount * this->colCount; i++) {
    result.getData()[i] = this->data[i] / divisor;
  }

  return result;
}

const bool Matrix::operator==(const Matrix &other) const {
  if (this->rowCount != other.rowCount || this->colCount != other.colCount) {
    return false;
  }

  bool result = true;

  for (int i = 0; i < this->rowCount; i++) {
    for (int j = 0; j < this->colCount; j++) {
      if ((*this)(i, j) != other(i, j)) {
        result = false;
      }
    }
  }

  return result;
}

Matrix Matrix::trim(size_t start, size_t end) const {
  if (start == end) {
    throw std::invalid_argument("Can't trim matrix, start and end equal!");
  }
  if (start >= this->rowCount || end >= this->rowCount) {
    throw std::invalid_argument(
        "Start or end is larger than matrix row count!");
  }
  Matrix result(end - start, this->colCount);

  for (int i = start; i < end; i++) {
    for (int j = 0; j < this->colCount; j++) {
      result(i - start, j) = (*this)(i, j);
    }
  }

  return result;
}

float Matrix::sum() const {
  float total = 0.f;

  for (size_t i = 0; i < this->data.size(); i++) {
    total += this->data[i];
  }

  return total;
}

Matrix Matrix::sumCols() const {
  Matrix result(1, this->colCount);

  for (int i = 0; i < this->colCount; i++) {
    float sum = 0.f;
    for (int j = 0; j < this->rowCount; j++) {
      sum += (*this)(j, i);
    }
    result(0, i) = sum;
  }

  return result;
}

const Matrix Matrix::one_hot_encode() const {
  if (this->colCount != 1) {
    throw std::invalid_argument("One Hot Encoding: No. of columns is " +
                                std::to_string(this->colCount) + "!\n");
  }
  Matrix result(this->rowCount, 10);
  for (int i = 0; i < this->rowCount; i++) {
    int idx = static_cast<int>((*this)(i, 0));
    result(i, idx) = 1.f;
  }
  return result;
}

const Matrix Matrix::argmax() const {
  Matrix result(this->rowCount, this->colCount);

  for (int i = 0; i < this->rowCount; i++) {
    float max = -999999.f;
    int idx = -9;
    for (int j = 0; j < this->colCount; j++) {
      if ((*this)(i, j) > max) {
        max = (*this)(i, j);
        idx = j;
      }
    }
    result(i, idx) = 1.f;
  }
  return result;
}

Matrix Matrix::transpose() const {
  Matrix result(this->colCount, this->rowCount);
  for (int i = 0; i < this->rowCount; i++) {
    for (int j = 0; j < this->colCount; j++) {
      result(j, i) = (*this)(i, j);
    }
  }

  return result;
}

void Matrix::setValue(size_t row, size_t column, float value) {
  this->data[row * this->colCount + column] = value;
}

void Matrix::randomise(float start, float end) {
  srand(time(nullptr));

     float range = end - start;
    size_t data_size = this->rowCount * this->colCount;

    for (size_t i = 0; i < data_size; ++i) {
        // Generate a random float within the range [start, end]
        this->data[i] = start + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / range);
    }
}

const std::string Matrix::format() const {
  std::string matrixFormat = "{\n";
  for (size_t i = 0; i < this->rowCount; i++) {
    matrixFormat += "(";
    for (size_t j = 0; j < this->colCount; j++) {
      matrixFormat += std::to_string((*this)(i, j));
      if (j != this->colCount - 1) {
        matrixFormat += ", ";
      }
    }
    matrixFormat += ")\n";
  }
  matrixFormat += "}";
  return matrixFormat;
}

const std::string Matrix::shape() const {
  std::string matrixShape = std::string("(") + std::to_string(this->rowCount) +
                            std::string(", ") + std::to_string(this->colCount) +
                            std::string(")");
  return matrixShape;
}

size_t Matrix::rows() const { return this->rowCount; }

size_t Matrix::cols() const { return this->colCount; }

const std::vector<float> &Matrix::getData() const { return this->data; }
std::vector<float> &Matrix::getData() { return this->data; }

Matrix::~Matrix() {}
