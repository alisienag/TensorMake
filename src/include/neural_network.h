#include "./layer.h"

#include <vector>

namespace Neural {
class Network {
 public:
    explicit Network(std::vector<size_t>&, std::vector<int>&);
    void useThreadCount(size_t threads);
    double train(Matrix&, Matrix&, float, int, int, bool);
    Matrix& feed(Matrix&);
    Matrix& getOutput();
 private:
    std::vector<Layer> layers;
    Matrix output;
    size_t user_thread_count;
};
}
