#include <lib/neuron.h>

int main()
{
    std::array<double, 1> input = {1};
    auto myNeuron = Neuron<1>(input, ActivationFunction::unipolar_func, ActivationFunction::bipolar_func_derivative, 1.0);

    std::cout << myNeuron.output(input) << std::endl;

    return 0;
}