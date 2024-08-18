#include "./data/mnist.hpp"
#include "./include/neural_network.h"
#include "include/layer.h"
#include "include/matrix.h"

int main() {
  std::vector<size_t> format = {784, 16, 12,  10};
  std::vector<int> activations = {SIGMOID_ID, SOFTMAX_ID, SOFTMAX_ID};

  Neural::Network neuralNetwork(format, activations);

  /*Matrix X(4, 2);
  X(0, 0) = 1.f;
  X(0, 1) = 1.f;
  X(1, 0) = 0.f;
  X(1, 1) = 0.f;
  X(2, 0) = 1.f;
  X(2, 1) = 0.f;
  X(3, 0) = 0.f;
  X(3, 1) = 1.f;

    Matrix Y(4, 1);
    Y(0, 0) = 0.f;
    Y(1, 0) = 0.f;
    Y(2, 0) = 1.f;
    Y(3, 0) = 1.f;*/

  auto dataset = data::mnist::getDataSet();

  Matrix X(dataset.training_images, true);
  Matrix Y(dataset.training_labels);
  Y = Y.transpose();
  Y = Y.one_hot_encode();

  std::cout << "One hot encoded!" << std::endl;

  neuralNetwork.train(X, Y, 0.1f, 50, MSE_LOSS);

  Matrix t_X(dataset.test_images, true);
  Matrix t_Y(dataset.test_labels);
  t_Y = t_Y.transpose();
  t_Y = t_Y.one_hot_encode();

  for (size_t i = 0; i < t_Y.rows() - 1; i++) {
    Matrix input = t_X.trim(i, i+1);
    print_matrix(input);
    Matrix output = t_Y.trim(i, i+1);
    print_matrix(output);
  }

  print_matrix(neuralNetwork.feed(X));
  return 0;
}
