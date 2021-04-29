#include <lib/neuron.h>

int main()
{
    std::array<double, 1> input = {1};
    auto myNeuron = Neuron<1>(input);

    std::cout << myNeuron.output(input) << std::endl;

    return 0;
}