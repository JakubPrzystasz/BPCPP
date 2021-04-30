#ifndef LAYER_H
#define LAYER_H
#include "includes.h"
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
    void feed(std::vector<double> &inputs, std::vector<double> &outputs)
    {
        for (size_t i{0}; i < neuron_count; i++)
            outputs[i] = neurons[i].feed(inputs);
    }

    /**
     * Constructor of single layer
    * As an argument it takes number of neurons, and number of neurons in previous layers
    * @arg n - number of neurons
    * @arg p - number of inputs
    * @arg m - number of previous weights to store
    */
    Layer(uint32_t n, uint32_t p, uint32_t m = 1, double learning_rate = 0.1, double momentum = 0.1)
    {
        this->neuron_count = n;
        this->momentum_count = m;
        this->input_count = p;
        this->learning_rate = learning_rate;
        this->momentum = momentum;
        for(uint32_t i{0};i<neuron_count;i++)
            neurons.push_back(Neuron(p,m));
    };

    /**
     * Destructor of single layer
     */
    ~Layer(){};
};

#endif