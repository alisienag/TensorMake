#pragma once

#include "../include/matrix.h"
#include "../mnist/mnist_reader.hpp"
#include "../mnist/mnist_utils.hpp"

namespace data {
  namespace mnist {
    inline auto getDataSet(){
      std::string PATH = "/home/Alisiena/Programming/cpp/TensorMake/TensorMake/src/mnist/";

      auto dataset = ::mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(PATH);
      
      return dataset;
    }
  }
}

