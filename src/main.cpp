#include "./include/matrix.h"

#include <time.h>

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

Matrix feed(Matrix& input, std::vector<Matrix>& weights, std::vector<Matrix>& bias, std::vector<Matrix>& results){
  Matrix hidden_first = input.parallel_mul(weights[0], 8, 64) + bias[0];
  Matrix activated_first = sigmoid(hidden_first);
  Matrix hidden_second = activated_first.parallel_mul(weights[1], 8, 64) + bias[1];
  Matrix activated_second = sigmoid(hidden_second);
  Matrix output = activated_second.parallel_mul(weights[2], 8, 64) + bias[2];
  Matrix result = sigmoid(output);

  results[0] = hidden_first;
  results[1] = hidden_second;
  results[2] = activated_first;
  results[3] = activated_second;
  results[INDEX_OUTPUT] = output;
  results[INDEX_RESULT] = result;

  return result;  
}

Matrix update_params(Matrix& input, Matrix& predicted, Matrix& result, std::vector<Matrix>& weights, std::vector<Matrix>& bias, std::vector<Matrix>& neurons, float learning_rate){
  Matrix d_loss_output = (result - predicted).dot(d_sigmoid(neurons[INDEX_OUTPUT])); //60000x10 
  Matrix d_loss_second = d_loss_output.parallel_mul(weights[2].transpose(), 8, 32).dot(d_sigmoid(neurons[INDEX_HIDDEN_SECOND]));
  Matrix d_loss_first = d_loss_second.parallel_mul(weights[1].transpose(), 8, 32).dot(d_sigmoid(neurons[INDEX_HIDDEN_FIRST]));
  
  Matrix d_loss_weights2 = neurons[INDEX_HIDDEN_SECOND+INDEX_ACTIVATED_OFFSET].transpose().parallel_mul(d_loss_output, 8, 32);
  Matrix d_loss_weights1 = neurons[INDEX_HIDDEN_FIRST+INDEX_ACTIVATED_OFFSET].transpose().parallel_mul(d_loss_second, 8, 32);
  Matrix d_loss_weights0 = input.transpose().parallel_mul(d_loss_first, 8, 32);
  
  weights[0] = weights[0] - d_loss_weights0.mul(learning_rate);
  weights[1] = weights[1] - d_loss_weights1.mul(learning_rate);
  weights[2] = weights[2] - d_loss_weights2.mul(learning_rate);
  
  //bias[0] = bias[0] - 

  return d_loss_output.dot(d_loss_output);
}

int main() {
  /*Matrix matrix(60000, 784);
  matrix.randomise(-0.5f, 0.5f);
  //matrix2.randomise(-0.5f, 0.5f);
  
  Matrix weight_1(784, 10); weight_1.randomise(-0.5f, 0.5f);
  Matrix weight_2(10, 12); weight_2.randomise(-0.9f, 0.95f);
  Matrix weight_3(12, 10); weight_3.randomise(-0.1f, 0.1f);
  Matrix bias_1(1, 10); bias_1.randomise(-0.5f, 0.5f);
  Matrix bias_2(1, 12); bias_2.randomise(-0.5f, 0.5);
  Matrix bias_3(1, 10); bias_3.randomise(-0.5f, 0.5f);

  std::vector<Matrix> weights = { weight_1, weight_2, weight_3 };
  std::vector<Matrix> bias = { bias_1, bias_2, bias_3 };
  std::vector<Matrix> results = {Matrix(1, 1), Matrix(1, 1), Matrix(1, 1),
                                 Matrix(1, 1), Matrix(1, 1), Matrix(1, 1)};
  Matrix result = feed(matrix, weights, bias, results);

  print_matrix(result);*/

  Matrix X(4, 2);
  X(0,0) = 0; X(0,1) = 0;
  X(1,0) = 1; X(1,1) = 1;
  X(2,0) = 0; X(2,1) = 1;
  X(3,0) = 1; X(3,1) = 0;

  Matrix Y(4,1);
  Y(0,0) = 0; Y(1,0) = 0; Y(2,0) = 1; Y(3,0) = 1;

  Matrix weight_1(2, 5); weight_1.randomise(-0.5f, 0.5f);
  Matrix weight_2(5, 5); weight_2.randomise(-0.5f, 0.5f);
  Matrix weight_3(5, 1); weight_3.randomise(-0.5f, 0.5f);
  std::vector<Matrix> weights; weights.push_back(weight_1); weights.push_back(weight_2); weights.push_back(weight_3);

  Matrix bias_1(1, 5); bias_1.randomise(-0.5f, 0.5f);
  Matrix bias_2(1, 5); bias_2.randomise(-0.5f, 0.5);
  Matrix bias_3(1, 1); bias_3.randomise(-0.5f, 0.5f);

  std::vector<Matrix> bias = { bias_1, bias_2, bias_3 };
  std::vector<Matrix> results = {Matrix(1, 1), Matrix(1, 1), Matrix(1, 1),
                                 Matrix(1, 1), Matrix(1, 1), Matrix(1, 1)};
  std::cout << "Starting to train!" << std::endl;
  
  for(int i = 0; i < 10000; i++){
    Matrix res = feed(X, weights, bias, results);
    update_params(X, Y, res, weights, bias, results, 0.1f);
  }

  print_matrix(feed(X, weights, bias, results));

  return 69;
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

