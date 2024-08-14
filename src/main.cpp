// #include "./data/mnist.hpp"
#include "./include/neural_network.h"
#include "include/layer.h"
#include "include/matrix.h"

int main() {
  /*    Layer layer(2, 5, RELU_ID);
      Matrix X(5, 2);
      X.randomise(-1.f, 1.f);
      print_matrix(layer.feed(X)); */

  std::vector<size_t> format = {2, 4, 4, 1};
  std::vector<int> activations = {SIGMOID_ID, SIGMOID_ID, SIGMOID_ID};

  Neural::Network neuralNetwork(format, activations);

  Matrix X(4, 2);
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
    Y(3, 0) = 1.f;

    print_matrix(X);
    print_matrix(Y);
  print_matrix(neuralNetwork.feed(X));

    neuralNetwork.train(X, Y, 0.5f, 10000, MSE_LOSS);

  print_matrix(neuralNetwork.feed(X));
  return 0;
}
