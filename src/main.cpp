#include "./data/mnist.hpp"
#include "./include/neural_network.h"
#include "include/layer.h"
#include "include/matrix.h"

#define NN_PRINT_STATUS 0

int main() {
  std::vector<size_t> format = {2, 4, 4, 1};
  std::vector<int> activations = {SIGMOID_ID, SIGMOID_ID, SIGMOID_ID};

  Neural::Network neuralNetwork(format, activations);
  neuralNetwork.useThreadCount(8);  // If i have 8 threads on my cpu

  Matrix x_test(4, 2);
  x_test(0, 0) = 0; x_test(0, 1) = 1;
  x_test(1, 0) = 1; x_test(1, 1) = 0;
  x_test(2, 0) = 0; x_test(2, 1) = 0;
  x_test(3, 0) = 1; x_test(3, 1) = 1;

  Matrix y_train(4, 1);
  y_train(0, 0) = 1;
  y_train(1, 0) = 1;
  y_train(2, 0) = 0;
  y_train(3, 0) = 0;

  print_matrix(neuralNetwork.feed(x_test));
  neuralNetwork.train(x_test, y_train, 0.1, 10000, MSE_LOSS, true);
  print_matrix(neuralNetwork.feed(x_test));
  return 0;
}
