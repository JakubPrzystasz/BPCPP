#include "layer.h"

Layer::Layer(uint32_t neurons, uint32_t inputs, double learning_rate, double momentum_const, rand_range &range, uint32_t batch_size)
{
    this->inputs = inputs;
    this->learning_rate = learning_rate;
    this->momentum_const = momentum_const;

    for (uint32_t i{0}; i < neurons; i++)
        this->neurons.push_back(Neuron(inputs, range, batch_size));

    this->batch_size = batch_size;
};
