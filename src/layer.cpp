#include "./include/layer.h"
#include <cmath>

Layer::Layer(size_t inputs, size_t outputs, size_t activation_id) {
    this->weights = Matrix(inputs, outputs);
    this->weights.randomise(-0.5f, 0.5f);
    this->bias = Matrix(1, outputs);
    this->bias.randomise(-0.5f, 0.5f);

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
    std::vector<float> max_row;
    std::vector<float> sum_row;

    for (int i = 0; i < input.rows(); i++) {
        sum_row.push_back(0.f);
        float max = -9999999.f;
        for (int j = 0; j < input.cols(); j++) {
            sum_row[i] += std::exp(input(i, j));
            if (max <= input(i, j)) {
                max = input(i, j);
            }
        }
        max_row.push_back(max);
    }

    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            std::cout << input(i, j) << std::endl;
            std::cout << max_row[i] << std::endl;
            result(i, j) = std::exp(input(i, j) - max_row[i]) / sum_row[i];
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
