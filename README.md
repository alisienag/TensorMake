
# TensorMake


A multi-layer perceptron framework written in pure C++. Allows for creating as many layers and neurons in each layer, with automatic training as long as correct input and desired outputs are given.
## Features

- Any number of neurons and layers
- Custom matrix class
- Easy to use
- Open source :D



## Installation

Install my-project with npm

```bash
  git clone https://github.com/alisienag/TensorMake/
  cd TensorMake
```
    
## Optimizations

Not many optimisations included. Support for using multiple threads for matrix multipicaton. (Yes I know using std::vector for my matrices are bad, planning to rewrite soon!)
## Running Tests

To run tests, run the following command

```bash
  cd TensorMake/src/build
  cmake .. ; make ;; ./myprogram
```
(If error is faced, delete all cmake related files except for CMakeLists.txt in "src")


## Usage/Examples

First include the "neural_network.h" header file.
```c++
#include "./include/neural_network.h"
```
Then you need to create an std::vector of your layers and neuron count

```c++
int main() {
    std::vector<size_t> format = {2, 4, 4, 1};
}
```

The value of each number states the amount of neurons in that layer. So in this example, we have an input layer of 2 neurons, two hidden layers with 4 neurons each, and finally an output layer of only 1 neuron.

Then we need to state which activation function to use for each layer (we don't use activation functions for the input layer)

```c++
int main() {
    std::vector<size_t> format = {2, 4, 4, 1};
    std::vector<int> activations = {SIGMOID_ID, SIGMOID_ID, SIGMOID_ID};
}
```

We have 3 options for activation layers:
- SIGMOID_ID
- RELU_ID
- SOFTMAX_ID

Please note, softmax way not work properly and sigmoid has been tested the most!

```c++
int main() {
    std::vector<size_t> format = {2, 4, 4, 1};
    std::vector<int> activations = {SIGMOID_ID, SIGMOID_ID, SIGMOID_ID};
}
```

We can now create our neural_network object.

```c++
int main() {
    std::vector<size_t> format = {2, 4, 4, 1};
    std::vector<int> activations = {SIGMOID_ID, SIGMOID_ID, SIGMOID_ID};

    Neural::Network neuralNetwork(format, activations);
    neuralNetwork.useThreadCount(8);  // If i have 8 threads on my cpu
}
```

We then create our input and desired output as matrices shown below:

```c++
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
```

Some of you may recognise this as the XOR function!

Now, we're going to print our original guess, and then the guess after training!

```C++
  print_matrix(neuralNetwork.feed(x_test));
  neuralNetwork.train(x_test, y_train, 0.1, 10000, MSE_LOSS, true);
  print_matrix(neuralNetwork.feed(x_test));
```

The function print_matrix prints any matrix passed into it to the console in a neat manner.

The feed function takes a matrix and input and returns the output.

The train function takes 6 parameters.

The first parameter x_test is the input data or matrix.

The second is the desired output data or matrix.

The third is the learning rate.

The fourth is the amount of epochs or training iterations.

The fifth is the type of loss to be used but this is currently a work in progress and wouldn't really change anything.

And finally the sixth being whether to print the loss for each iteration.

Finally this is the full code:

```c++
#include "./include/neural_network.h"

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
  neuralNetwork.train(x_test, y_train, 0.1, 10000, MSE_LOSS, false);
  print_matrix(neuralNetwork.feed(x_test));
  return 0;
}
```

This is the output I got running this code:

```bash
{
(0.553602)
(0.555120)
(0.553624)
(0.519932)
}
{
(0.995292)
(0.995469)
(0.000628)
(0.008766)
}
```
