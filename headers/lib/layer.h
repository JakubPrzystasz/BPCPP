#pragma once 

#include "neuron.h"

class Layer
{
public:
    uint32_t neuron_count;
    uint32_t input_count;
    uint32_t momentum_count;

    /**
     * Learning rate for layer
     */
    double learning_rate;

    /**
     * Momentum for layer
     */
    double momentum;

    /**
     * Container for neurons 
     */
    std::vector<Neuron> neurons;

    /**
     * Feed layer with data
     * @arg inputs - vector of input values
     * @arg outputs - vector of outputs values
     */
    inline void feed(std::vector<double> &inputs, std::vector<double> &outputs, std::vector<double> &input_values){
        for (size_t i{0}; i < neuron_count; i++)
            outputs[i] = neurons[i].feed(inputs,input_values[i]);
    }

    /**
     * Constructor of single layer
    * As an argument it takes number of neurons, and number of neurons in previous layers
    * @arg n - number of neurons
    * @arg p - number of inputs
    * @arg m - number of previous weights to store
    */
    Layer(uint32_t n, uint32_t p, uint32_t m = 1, double learning_rate = 0.1, double momentum = 0.1);

    /**
     * Destructor of single layer
     */
    ~Layer(){};
};
