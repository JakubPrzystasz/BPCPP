#include "layer.h"

Layer::Layer(uint32_t n, uint32_t p, uint32_t m, double learning_rate, double momentum)
{
    this->neuron_count = n;
    this->momentum_count = m;
    this->input_count = p;
    this->learning_rate = learning_rate;
    this->momentum = momentum;
    for (uint32_t i{0}; i < neuron_count; i++)
        neurons.push_back(Neuron(p, m));
};