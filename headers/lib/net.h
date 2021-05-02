#pragma once

#include "layer.h"

class Net
{
    /**
     * Container for output of each layer
     */
    data_set output;

    /**
     * Cost vector
     */
    data_row cost;
    uint32_t layers_count;

public:
    std::vector<Layer> layers;

    data_set input;
    data_set target;

    /**
     *  Feed net with given data
     */
    inline void feed(data_row &input)
    {
        this->layers[0].feed(input, this->output[0]);

        for (uint32_t i{1}; i < layers.size(); i++)
            this->layers[i].feed(this->output[i - 1], this->output[i]);
    }

    /**
     * Add layer to net
     */
    void add_layer(uint32_t neurons, double learning_rate = 0.1, double momentum = 0.1, size_t prev_weights = 1);

    /**
     *  Constructor of neural network
     *  @arg input - input data set
     *  @arg target - target data set
     */
    Net(data_set &input, data_set &target);

    ~Net();
};
