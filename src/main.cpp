#include "./include/matrix.h"
#include "./data/mnist.hpp"

#include <time.h>
#include <algorithm>

#define TEST_TIME(x) start = clock(); x; end = clock(); std::cout << std::cout << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
#define INDEX_HIDDEN_FIRST 0
#define INDEX_HIDDEN_SECOND 1
#define INDEX_ACTIVATED_OFFSET 2
#define INDEX_OUTPUT 4
#define INDEX_RESULT 5

#define sep() std::cout << "1" << std::endl;

Matrix softmax(Matrix&);
Matrix reLU(Matrix&);
Matrix sigmoid(Matrix&);
Matrix d_reLU(Matrix&);
Matrix d_sigmoid(Matrix&);
float accuracy(const Matrix& result, const Matrix& expected);

Matrix feed(Matrix& input, std::vector<Matrix>& weights, std::vector<Matrix>& bias, std::vector<Matrix>& results){
  Matrix hidden_first = input.parallel_mul(weights[0], 16, 64) + bias[0];
  Matrix activated_first = reLU(hidden_first);
  Matrix hidden_second = activated_first.parallel_mul(weights[1], 16, 64) + bias[1];
  Matrix activated_second = reLU(hidden_second);
  Matrix output = activated_second.parallel_mul(weights[2], 16, 64) + bias[2];
  Matrix result = softmax(output);

  results[0] = hidden_first;
  results[1] = hidden_second;
  results[2] = activated_first;
  results[3] = activated_second;
  results[INDEX_OUTPUT] = output;
  results[INDEX_RESULT] = result;
  return result;
}

double crossEntropyLoss(const std::vector<float>& predicted, const std::vector<float>& labels) {

    double loss = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        // Ensure probabilities are clipped to avoid log(0)
        double prob = std::max(std::min(predicted[i], static_cast<float>(1.0 - 1e-15)), static_cast<float>(1e-15));
        if (labels[i] == 1) {
            loss -= std::log(prob);
        } else {
            loss -= std::log(1.0 - prob);
        }
    }

    return loss / predicted.size();
}

Matrix update_params(const Matrix &input, const Matrix &predicted, const Matrix &result,
                     std::vector<Matrix> &weights, std::vector<Matrix> &bias,
                     std::vector<Matrix> &neurons, const float learning_rate) {
  bool softmax = true;
  Matrix d_loss_output(1, 1);
  if (softmax) {
    d_loss_output = result - predicted;
  } else {
    d_loss_output =
        (result - predicted).dot(d_sigmoid(neurons[INDEX_OUTPUT])); // 60000x10
  }

  //std::cout << "Loss: " << crossEntropyLoss(predicted.getData(), result.getData()) << std::endl;
    // d_loss_output = (result - predicted).dot(d_sigmoid(neurons[INDEX_OUTPUT]));
  // //60000x10
  Matrix d_loss_second =
      d_loss_output.parallel_mul(weights[2].transpose(), 16, 16)
          .dot(d_reLU(neurons[INDEX_HIDDEN_SECOND]));
  Matrix d_loss_first =
      d_loss_second.parallel_mul(weights[1].transpose(), 16, 16)
          .dot(d_reLU(neurons[INDEX_HIDDEN_FIRST]));

  Matrix d_loss_weights2 = neurons[INDEX_HIDDEN_SECOND + INDEX_ACTIVATED_OFFSET]
                               .transpose()
                               .parallel_mul(d_loss_output, 16, 16);
  Matrix d_loss_weights1 = neurons[INDEX_HIDDEN_FIRST + INDEX_ACTIVATED_OFFSET]
                               .transpose()
                               .parallel_mul(d_loss_second, 16, 16);
  Matrix d_loss_weights0 = input.transpose().parallel_mul(d_loss_first, 16, 16);

  weights[0] = weights[0] - d_loss_weights0.mul(learning_rate);
  weights[1] = weights[1] - d_loss_weights1.mul(learning_rate);
  weights[2] = weights[2] - d_loss_weights2.mul(learning_rate);

  bias[0] = bias[0] - d_loss_first.sumCols().mul(learning_rate);
  bias[1] = bias[1] - d_loss_second.sumCols().mul(learning_rate);
  bias[2] = bias[2] - d_loss_output.sumCols().mul(learning_rate);

  return d_loss_output.dot(d_loss_output);
}

int main() {
  /*Matrix X(4, 2);
  X(0, 0) = 0;
  X(0, 1) = 0;
  X(1, 0) = 1;
  X(1, 1) = 1;
  X(2, 0) = 0;
  X(2, 1) = 1;
  X(3, 0) = 1;
  X(3, 1) = 0;*/

  /*Matrix Y(4, 2);
  Y(0, 0) = 1;
  Y(0, 1) = 0;
  Y(1, 0) = 1;
  Y(1, 1) = 0;
  Y(2, 0) = 0;
  Y(2, 1) = 1;
  Y(3, 0) = 0;
  Y(3, 1) = 1;*/

  auto dataset = data::mnist::getDataSet();

  Matrix mnist_training_input(dataset.training_images);
  Matrix mnist_training_labels(dataset.training_labels);
  Matrix mnist_testing_input(dataset.test_images);
  Matrix mnist_testing_labels(dataset.test_labels);
  mnist_training_labels = mnist_training_labels.transpose().one_hot_encode();
  mnist_testing_labels = mnist_testing_labels.transpose().one_hot_encode();

  Matrix weight_1(784, 32); weight_1.randomise(-0.5f, 0.5f);
  Matrix weight_2(32, 16); weight_2.randomise(-0.5f, 0.5f);
  Matrix weight_3(16, 10); weight_3.randomise(-0.5f, 0.5f);
  std::vector<Matrix> weights; weights.push_back(weight_1); weights.push_back(weight_2); weights.push_back(weight_3);

  Matrix bias_1(1, 32); bias_1.randomise(-0.5f, 0.5f);
  Matrix bias_2(1, 16); bias_2.randomise(-0.5f, 0.5);
  Matrix bias_3(1, 10); bias_3.randomise(-0.5f, 0.5f);

  std::vector<Matrix> bias = { bias_1, bias_2, bias_3 };
  std::vector<Matrix> results = {Matrix(1, 1), Matrix(1, 1), Matrix(1, 1),
                                 Matrix(1, 1), Matrix(1, 1), Matrix(1, 1)};
  Matrix trim = mnist_testing_input.trim(0,1);
  Matrix res = feed(trim, weights, bias, results);
  float learning_rate = 0.00001f;
  for(int i = 0; i < 50; i++){
    Matrix res = feed(mnist_training_input, weights, bias, results);
    update_params(mnist_training_input, mnist_training_labels, res, weights, bias, results, learning_rate);
  }
  print_matrix(res);
  print_matrix(mnist_testing_labels.trim(0, 1));
  res = feed(trim, weights, bias, results);
  print_matrix(res);
  print_matrix(mnist_testing_labels.trim(0, 1));

  for(int i = 0; i < mnist_testing_input.rows(); i++){
    trim = mnist_testing_input.trim(i, i+1);
    Matrix trim_act = mnist_testing_labels.trim(i, i+1);
    res = feed(trim, weights, bias, results);
    int maxcols = 0;
    int actual = 0;
    for(int i = 0; i < res.cols(); i++){
      if(res(0, i) == 1){
        maxcols = i;
      }
      if(trim_act(0,i) == 1){
        actual = i;
      }
    }
    std::cout << "Prediction: " << maxcols << std::endl;
    std::cout << "ACTUAL: " << actual << std::endl;
    sleep(1);
  }



  //print_matrix(feed(X, weights, bias, results));
  //
  std::cout << "Post-training Accuracy: " << accuracy(feed(mnist_testing_input, weights, bias, results).argmax(), mnist_testing_labels) << std::endl;

  return 69;
}

float accuracy(const Matrix& result, const Matrix& expected) {
  float accuracy = 0.f;
  for (int i = 0; i < result.rows(); i++){
    for(int j = 0; j < result.cols(); j++){
      if(result(i,j) == 1){
        if(expected(i,j) == 1){
          accuracy += 1;
          break;
        }
      }else if(result(i, j) == 0){
        if(expected(i, j) == 1) {
          break;
        }
      }
    }
  }

  accuracy /= result.rows();

  return accuracy * 100;
}

Matrix softmax(Matrix& x) {
  Matrix result(x.rows(), x.cols());
  for (int i = 0; i < x.rows(); i++) {
    float max = x(i, 0);
    for (int j = 1; j < x.cols(); j++) {
      if (x(i, j) > max) {
        max = x(i, j);
      }
    }
    float sum = 0.f;
    for (int j = 0; j < x.cols(); j++) {
      result(i, j) = std::exp(x(i, j) - max);
      sum += result(i, j);
    }
    for (int j = 0; j < x.cols(); j++) {
      result(i, j) /= sum;
    }
  }
  return result;
}

Matrix reLU(Matrix& x){
  Matrix result(x.rows(), x.cols());

  for(int i = 0; i < x.rows() * x.cols(); i++){
    if(x.getData()[i] > 0){
      result.getData()[i] = x.getData()[i];
    }
  }

  return result;
}

Matrix sigmoid(Matrix& x){
  Matrix result(x.rows(), x.cols());

  for(int i = 0; i < x.rows() * x.cols(); i++){
    result.getData()[i] = float(1.f / (1.f + std::exp(-x.getData()[i])));
  }

  return result;
}

Matrix d_sigmoid(Matrix& x){
  Matrix result(x.rows(), x.cols());

  for(int i = 0; i < x.rows() * x.cols(); i++){
    float sig = float(1.f / (1.f + std::exp(-x.getData()[i])));
    result.getData()[i] =  (sig * (1 - sig));
  }
  return result;
}

Matrix d_reLU(Matrix& x){
  Matrix result(x.rows(), x.cols());

  for(int i = 0; i < x.rows() * x.cols(); i++){
    if(x.getData()[i] > 0){
      result.getData()[i] = 1;
    }
  }

  return result;
}

