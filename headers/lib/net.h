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

    void __add_layer(uint32_t neurons, double learning_rate, double momentum, size_t prev_weights);

public:
    double learning_rate;
    double momentum;
    uint32_t prev_weights;

    std::vector<Layer> layers;

    data_set input;
    data_set target;

    /**
     * Returns cost of single input data(SSE)
     */
    inline double get_cost(uint32_t row_num)
    {
        auto input_row = this->input[row_num];
        auto target_row = this->target[row_num];
        double cost_sum = 0;

        //Feed net with data
        this->feed(input_row);
        //Calculate cost for each output neuron
        auto output_row = this->output[this->layers_count - 1];

        for (uint32_t i{0}; i < this->output[layers_count - 1].size(); i++)
        {
            this->cost[i] = target_row[i] - output_row[i];
            this->cost[i] = this->cost[i] * this->cost[i];
            cost_sum += this->cost[i];
        }

        return cost_sum * 0.5;
    }

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
    void add_layer(uint32_t neurons, double learning_rate, double momentum, size_t prev_weights);

    void add_layer(uint32_t neurons);

    /**
     *  Constructor of neural network
     *  @arg input - input data set
     *  @arg target - target data set
     */
    Net(data_set &input, data_set &target, double learning_rate = 0.1, double momentum = 0.1, size_t prev_weights = 1);

    ~Net();
};
