#include "./include/matrix.h"

Matrix::Matrix(size_t rows, size_t columns) {
  this->rowCount = rows;
  this->colCount = columns;
  //this->data = reinterpret_cast<float*>(malloc(sizeof(float) * rows * columns));
  this->data = new float[rows * columns];
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      this->data[i * columns + j] = 0.f;
    }
  }

}

Matrix Matrix::dot(const Matrix &other) const {
  if (this->rows() != other.rows() || this->cols() != other.cols()) {
    throw std::invalid_argument("Matrix 'dot': Matrix Shapes not equal!");
  }
  Matrix result(this->rowCount, this->colCount);
  for (int i = 0; i < this->rowCount * this->colCount; i++) {
    result.getData()[i] = this->data[i] * other.getData()[i];
  }
  return result;
}

Matrix Matrix::mul(const Matrix &other) const {
  int block_size = 256;

  if (this->colCount != other.rows()) {
    std::string error_message = "Matrix multiplication: " + this->shape() +
                                " != " + other.shape() + "!";
    throw std::invalid_argument(error_message);
  }
  Matrix result(this->rowCount, other.cols());
  int n = this->rowCount;
  int m = this->colCount;
  int p = other.cols();

  for (int ii = 0; ii < n; ii += block_size) {
    for (int jj = 0; jj < p; jj += block_size) {
      for (int kk = 0; kk < m; kk += block_size) {
        for (int i = ii; i < std::min(ii+block_size, n); i++) {
          for (int j = jj; j < std::min(jj+block_size, p); j++) {
            for (int k = kk; k < std::min(kk+block_size, m); k++) {
              result[i][j] += (*this)[i][k] * other[k][j];
            }
          }
        }
      }
    }
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
        sum += (*this)[i][k] * other[k][j];
      }
      result[i][j] = sum;
    }
  }

  return result;
}


float *Matrix::operator[](size_t rows) const {
  if (rows >= this->rowCount) {
    throw std::invalid_argument(
        "Matrix indexing: Row indexed is greater than row count!");
  }
  return reinterpret_cast<float *>(this->data + (this->colCount * rows));
}

Matrix Matrix::operator+(const Matrix &other) const {
  Matrix result(this->rowCount, this->colCount);
  if (other.rows() != this->rows() || other.cols() != this->cols()) {
    throw std::invalid_argument("Matrix addition: Dimensions dont match!");
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
    throw std::invalid_argument("Matrix subtraction: Dimensions dont match!");
  } else {
    for (int i = 0; i < this->rowCount * this->colCount; i++) {
      result.getData()[i] = this->data[i] - other.getData()[i];
    }
  }

  return result;
}

Matrix Matrix::transpose() const {
  Matrix result(this->colCount, this->rowCount);
  for (int i = 0; i < this->colCount; i++) {
    for (int j = 0; j < this->rowCount; j++) {
      result[j][i] = (*this)[i][j];
    }
  }

  return result;
}

void Matrix::setValue(size_t row, size_t column, float value) {
  this->data[row * this->rowCount + column] = value;
}

void Matrix::randomise(float start, float end) {
  srand(time(nullptr));

  float total = std::abs(start) + std::abs(end);

  size_t data_size = this->rowCount * this->colCount;

  for (size_t i = 0; i < data_size; i++) {
    this->data[i] = start + ((static_cast<float>(rand())/RAND_MAX) * total);
  }
}

const std::string Matrix::format() const {
  std::string matrixFormat = "{\n";
  for (size_t i = 0; i < this->rowCount; i++) {
    matrixFormat += "(";
    for (size_t j = 0; j < this->colCount; j++) {
      matrixFormat += std::to_string(this->data[i * rowCount + j]);
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

float *Matrix::getData() const { return this->data; }

Matrix::~Matrix() { delete this->data; }
