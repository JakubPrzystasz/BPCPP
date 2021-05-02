#ifndef NET_H
#define NET_H

#include "layer.h"
#include "includes.h"

typedef std::vector<double> data_row;
typedef std::vector<data_row> data_set;

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
    void feed(data_row &input)
    {
        this->layers[0].feed(input, this->output[0]);

        for (uint32_t i{1}; i < layers.size(); i++)
            this->layers[i].feed(this->output[i-1],this->output[i]);
    }


    /**
     * Add layer to net
     */
    void add_layer(uint32_t neurons, double learning_rate = 0.1, double momentum = 0.1, size_t prev_weights = 1)
    {
        if (this->layers.size() < 1)
            this->layers.push_back(Layer(neurons, neurons, prev_weights, learning_rate, momentum));
        else
            this->layers.push_back(Layer(neurons, this->layers.back().neuron_count, prev_weights, learning_rate, momentum));

        //Setup helper containers
        this->cost = data_row(this->layers.back().neuron_count);
        this->output.erase(this->output.begin(),this->output.end());
        for (auto &layer : layers)
            this->output.push_back(data_row(layer.neuron_count));
        
        this->layers_count += 1;
    }

    /**
     *  Constructor of neural network
     *  @arg input - input data set
     *  @arg target - target data set
     */
    Net(data_set &input, data_set &target)
    {
        this->input = input;
        this->target = target;
        this->cost = data_row(target[0].size(), 0);
    }

    ~Net() {}
};

#endif