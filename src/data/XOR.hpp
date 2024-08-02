#pragma once

#include "../include/matrix.h"

namespace data {
  namespace XOR {
    inline matrix getTrainingInputs(){
      matrix X(4, 2, false);
	    std::vector<float> data1 = { 1, 1 }; X.setRow(data1, 0);
	    std::vector<float> data2 = { 0, 0 }; X.setRow(data2, 1);
	    std::vector<float> data3 = { 1, 0 }; X.setRow(data3, 2);
	    std::vector<float> data4 = { 0, 1 }; X.setRow(data4, 3);
      return X;
    }

    inline matrix getTrainingOutputs(){
      matrix Y(1, 4, false);
	    std::vector<float> data5 = { 0, 0, 1, 1};
	    Y.setRow(data5, 0);
      Y = Y.transpose();
      return Y;
    }
  }
}
