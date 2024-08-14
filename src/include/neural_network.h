#include "./layer.h"

#include <vector>

namespace Neural {
class Network {
 public:
    explicit Network(std::vector<size_t>&, std::vector<int>&);
    double train(Matrix&, Matrix&, float, int, int);
    Matrix& feed(Matrix&);
    

    Matrix& getOutput();
 private:
    std::vector<Layer> layers;

    Matrix output;
};
}
