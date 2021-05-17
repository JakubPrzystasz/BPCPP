#include "layer.h"

Layer::Layer(uint32_t neurons, uint32_t inputs, LearnParams params)
{
    this->learn_parameters = params;

    this->inputs = inputs;

    this->neurons = std::vector<Neuron>(neurons, Neuron(inputs, this->learn_parameters));
}
