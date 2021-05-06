#include "layer.h"

Layer::Layer(uint32_t neurons, uint32_t inputs, double learning_rate, double momentum_const, double rand_min, double rand_max)
{
    this->inputs = inputs;
    this->learning_rate = learning_rate;
    this->momentum_const = momentum_const;

    for (uint32_t i{0}; i < neurons; i++)
        this->neurons.push_back(Neuron(inputs, rand_min, rand_max));
};
