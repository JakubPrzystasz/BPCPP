#include <lib/neuron.h>

int main()
{
    std::array<double, 2> input = {1,2};
    auto myNeuron = Neuron<2>();

    std::cout << myNeuron.output(input) << std::endl;

    return 0;
}