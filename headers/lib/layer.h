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
     * Size of batch
     */
    uint32_t batch_size;

    /**
     * Container for neurons 
     */
    std::vector<Neuron> neurons;

    /**
     * Constructor of single layer
    * As an argument it takes number of neurons, and number of neurons in previous layers
    * @arg neurons - number of neurons
    * @arg inputs - number of inputs
    * @arg learning_rate
    * @arg momentum_const
    * @arg range - pair of double, first is min, second is max - defines range for weights and biases initailization random values
    * @arg batch_size 
    */
    Layer(uint32_t neurons, uint32_t inputs, double learning_rate, double momentum_const, rand_range &range, uint32_t batch_size = 1);

    ~Layer(){};

};
