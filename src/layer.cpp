#include "./include/layer.h"
#include <cmath>

Layer::Layer(size_t inputs, size_t outputs, size_t activation_id) {
    this->weights = Matrix(inputs, outputs);
    this->weights.randomise(-0.3f, 0.3f);
    this->bias = Matrix(1, outputs);

    this->activation_id = activation_id;
}

Matrix &Layer::feed(Matrix &input) {
    this->hidden = input.parallel_mul(this->weights, 4, 32) + this->bias;
    this->activated = this->activation(this->hidden);
    return this->activated;
}

Matrix Layer::activation(Matrix& input) const {
    if (this->activation_id == SIGMOID_ID) {
        return this->sigmoid(input);
    } else if (this->activation_id == RELU_ID) {
        return this->reLU(input);
    } else if (this->activation_id == SOFTMAX_ID) {
        return this->softmax(input);
    } else {
        return Matrix(313, 313);
    }
}


Matrix Layer::derivative() const {
    if (this->activation_id == SIGMOID_ID) {
        return this->d_sigmoid(this->activated);
    } else if (this->activation_id == RELU_ID) {
        return this->d_reLU(this->hidden);
    } else if (this->activation_id == SOFTMAX_ID) {
        return this->softmax(this->activated);
    } else {
        return Matrix(313, 313);
    }
}

Matrix& Layer::getHidden() {
    return this->hidden;
}

Matrix& Layer::getActivated() {
    return this->activated;
}

Matrix& Layer::getWeights() {
    return this->weights;
}

Matrix& Layer::getBias() {
    return this->bias;
}

Matrix Layer::softmax(const Matrix& input) const {
    Matrix result(input.rows(), input.cols());
    
    std::vector<float> sum_of_rows;

    for (size_t i = 0; i < input.rows(); i++) {
      float sum = 0.f;
      for (size_t j = 0; j < input.cols(); j++) {
        sum += std::exp(input(i, j));
      }
      sum_of_rows.push_back(sum);

      for (size_t j = 0; j < input.cols(); j++) {
        result(i, j) = std::exp(input(i, j)) / sum_of_rows[i];
      }
    }
    return result;
}

Matrix Layer::reLU(const Matrix &input) const {
  Matrix result(input.rows(), input.cols());
  for (int i = 0; i < input.cols() * input.rows(); i++) {
    if (input.getData()[i] > 0) {
      result.getData()[i] = input.getData()[i];
    } else {
      // do nothing as it is already zero :D
    }
  }
  return result;
}
Matrix Layer::sigmoid(const Matrix &input) const {
  Matrix result(input.rows(), input.cols());
  for (int i = 0; i < input.cols() * input.rows(); i++) {
    result.getData()[i] = 1.f / (1.f + std::exp(-1 * input.getData()[i]));
  }
  return result;
}
Matrix Layer::d_reLU(const Matrix &input) const {
  Matrix result(input.rows(), input.cols());
  for (int i = 0; i < input.cols() * input.rows(); i++) {
    if (input.getData()[i] > 0) {
      result.getData()[i] = 1;
    } else {
      // do nothing as it is already zero :D
    }
  }
  return result;
}
Matrix Layer::d_sigmoid(const Matrix &input) const {
  Matrix result(input.rows(), input.cols());
  for (int i = 0; i < input.cols() * input.rows(); i++) {
    float exp = input.getData()[i];
    result.getData()[i] = exp * (1.f - exp);
  }
  return result;
}
int Layer::getActivationID() {
  return this->activation_id;
}
