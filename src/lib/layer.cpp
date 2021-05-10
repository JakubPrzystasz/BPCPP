#include "layer.h"

Layer::Layer(uint32_t neurons, uint32_t inputs, LearnParams params)
{
    this->learn_parameters = params;

    this->inputs = inputs;

    for (uint32_t i{0}; i < neurons; i++)
        this->neurons.push_back(Neuron(inputs, this->learn_parameters));

};
