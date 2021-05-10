#pragma once 

#include "neuron.h"

struct Layer
{

	/**
	 * Container for learning params, for all neurons in layer
     * it is kind of blue print for making new neurons in that layer
	 */
	LearnParams learn_parameters;

    /**
     * Number of neurons in previous layer
     */
    uint32_t inputs;

    /**
     * Container for neurons 
     */
    std::vector<Neuron> neurons;

    /**
     * Constructor of single layer
    * As an argument it takes number of neurons, and number of neurons in previous layers
    * @arg neurons - number of neurons
    * @arg inputs - number of inputs (neurons in previous layer)
    * @arg params - learning parameters for each neuron 
    */
    Layer(uint32_t neurons, uint32_t inputs, LearnParams params);

    ~Layer(){};

};
