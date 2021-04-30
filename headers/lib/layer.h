#ifndef LAYER_H
#define LAYER_H
#include "includes.h"
#include "neuron.h"

/**
 * As an argument it takes number of neurons, and number of neurons in previous layers
 * @arg n - number of neurons
 * @arg p - number of neurons in previous layer
 * @arg m - number of previous + 1 weights to store
 */
template <size_t n, size_t p, size_t m>
class Layer
{

public:
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
    std::array<Neuron<p, m>, n> neurons;

    /**
     * Feed layer with data
     * @arg inputs - vector of input values
     * @arg outputs - vector of outputs values
     */
    void feed(std::array<double, p> &inputs, std::array<double, n> &outputs)
    {
        for (size_t i{0}; i < n; i++)
            outputs[i] = neurons[i].feed(inputs);
    }

    /**
     * Constructor of single layer
     */
    Layer(double learning_rate = 0.1, double momentum = 0.1)
    {
        this->learning_rate = learning_rate;
        this->momentum = momentum;
    };

    /**
     * Destructor of single layer
     */
    ~Layer(){};
};

#endif