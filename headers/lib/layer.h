#pragma once 

#include "neuron.h"

class Layer
{
public:
    /**
     * Number of neurons in previous layer
     */
    uint32_t inputs;

    /**
     * Learning rate for layer
     */
    double learning_rate;

    /**
     * Momentum for layer
     */
    double momentum_const;

    /**
     * Container for neurons 
     */
    std::vector<Neuron> neurons;

    /**
     * Constructor of single layer
    * As an argument it takes number of neurons, and number of neurons in previous layers
    * @arg neurons - number of neurons
    * @arg inputs - number of inputs
    */
    Layer(uint32_t neurons, uint32_t inputs, double learning_rate, double momentum_const, double rand_min = -1.0, double rand_max = 1.0);

    ~Layer(){};

};
