#include "layer.h"

Layer::Layer(uint32_t neurons, uint32_t inputs, double learning_rate, double momentum_const, uint32_t ID, Layer* prev_layer)
{
    this->prev_layer = prev_layer;
    this->ID = ID;
    this->inputs = inputs;
    this->learning_rate = learning_rate;
    this->momentum_const = momentum_const;

    for (uint32_t i{0}; i < neurons; i++)
        this->neurons.push_back(Neuron(inputs, i, prev_layer));
};