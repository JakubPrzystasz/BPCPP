#include <lib/layer.h>

int main()
{
    auto myLayer = Layer<4, 4, 1>();
    std::array<double, 4> input = {0.4, 0.3, 0.1, 0.83};
    std::array<double, 4> output;
    myLayer.feed(input, output);

    for(double &val : output)
        std::cout << val << std::endl;

    return 0;
}