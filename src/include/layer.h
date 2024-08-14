#pragma once

#define SIGMOID_ID 0
#define RELU_ID 1
#define SOFTMAX_ID 2

#define MSE_LOSS 0

#include "./matrix.h"

class Layer {
 public:
	Layer(size_t inputs, size_t outputs, size_t activation_id);

    Matrix& feed(Matrix&);

    Matrix activation(Matrix&) const;
    Matrix derivative() const;

    Matrix& getHidden();
    Matrix& getActivated();

    Matrix& getWeights();
    Matrix& getBias() ;
 private:
    Matrix softmax(const Matrix&) const;
    Matrix sigmoid(const Matrix&) const;
    Matrix reLU(const Matrix&) const;

    Matrix d_sigmoid(const Matrix&) const;
    Matrix d_reLU(const Matrix&) const;

	Matrix weights;
    Matrix bias;
	Matrix hidden;
	Matrix activated;

    int activation_id = 0;  // default is sigmoid
};
