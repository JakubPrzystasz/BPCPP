#ifndef LAYER_H
#define LAYER_H
#include "includes.h"
#include "neuron.h"

/**
 * As an argument it takes number of neurons, and number of neurons in previous layers
 * @arg n - number of neurons
 * @arg m - numbet of neurons in previous layer
 */
template<size_t n, size_t m>
class Layer{

public:

    std::array<Neuron<m>, n> neurons;

    Layer(){};

    ~Layer(){};

};

#endif