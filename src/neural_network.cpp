#include "./include/neural_network.h"
#include "include/layer.h"
#include "include/matrix.h"
#include <cmath>
#include <cstddef>


Neural::Network::Network(std::vector<size_t> &format, std::vector<int> &activation_id) {
  this->user_thread_count = 1;
  for (size_t i = 0; i < format.size() - 1; i++) {
    Layer layer(format[i], format[i + 1], activation_id[i]);
    this->layers.push_back(layer);
  }
}

void Neural::Network::useThreadCount(size_t threads) {
  this->user_thread_count = threads;
}

double Neural::Network::train(Matrix &X, Matrix &Y, float learning_rate, int iters, int loss_function, bool print_status) {
  for (int i = 0; i < iters; i++) {
    this->feed(X);
    if (print_status) {
      Matrix LOSS_A = (this->output - Y);
      LOSS_A = LOSS_A.dot(LOSS_A);
      float loss = LOSS_A.sum() / X.rows();
      std::cout << "Iter: " << i << " Loss: " << loss << std::endl;
    }
    Matrix d_loss_output = (this->output - Y) / X.rows();
    std::vector<Matrix> d_hiddens;
    std::vector<Matrix> d_weights;
    for (size_t i = 0; i < this->layers.size(); i++) {
      d_hiddens.push_back(Matrix(1, 1));
      d_weights.push_back(Matrix(1, 1));
    }
    d_hiddens[d_hiddens.size() - 1] = d_loss_output;
    d_weights[d_weights.size() - 1] =
        this->layers[this->layers.size() - 2]
            .getActivated()
            .transpose()
            .parallel_mul(d_hiddens[d_hiddens.size() - 1],
                          this->user_thread_count, 32);

    for (size_t i = this->layers.size() - 2; i > 0; i--) {
        d_hiddens[i] = d_hiddens[i+1].parallel_mul(
        this->layers[i+1].getWeights().transpose(), 4,
        32).dot(this->layers[i].derivative());
      d_weights[i] = this->layers[i-1].getActivated().transpose().parallel_mul(
          d_hiddens[i], this->user_thread_count, 32);
    }
    d_hiddens[0] =
        d_hiddens[1]
            .parallel_mul(this->layers[1].getWeights().transpose(),
                          this->user_thread_count, 32)
            .dot(this->layers[0].derivative());
    d_weights[0] = X.transpose().parallel_mul(d_hiddens[0],
                                              this->user_thread_count, 32);
    for (int i = 0; i < this->layers.size(); i++) {
      this->layers[i].getWeights() =
          this->layers[i].getWeights() - (d_weights[i].mul(learning_rate));
      this->layers[i].getBias() = this->layers[i].getBias() -
                                  (d_hiddens[i].sumCols().mul(learning_rate));
    }
  }

  return 1.0;
}

Matrix &Neural::Network::feed(Matrix &input) {
  Matrix intermediate = this->layers[0].feed(input);
  for (size_t i = 1; i < this->layers.size(); i++) {
    intermediate = this->layers[i].feed(intermediate);
  }

  this->output = intermediate;

  return this->output;
}
