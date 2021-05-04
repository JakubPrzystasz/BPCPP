#pragma once

#include "layer.h"

class Net
{
    uint32_t layers_count;

    /**
     * Container for output of each neuron in each layer
     */
    data_set output;

    /**
     * Container for inputs for each neuron in each layer
     */
    data_set neurons_inputs;

    /**
     * Cost vector
     */
    data_row cost;

    /**
     * Size of batch
     */
    uint32_t batch_size;

    void __add_layer(uint32_t neurons, double learning_rate, double momentum, size_t prev_weights);

    inline double __get_delta(uint32_t layer_num, uint32_t neuron_num, uint32_t origin_layer, uint32_t origin_neuron)
    {
        if (layer_num == this->layers_count - 1)
        {
            auto neuron = this->layers[layer_num].neurons[neuron_num];
            double weight = 1.0;
            if (layer_num != origin_layer)
                weight = neuron.weights[origin_neuron];

            return (weight * this->cost[neuron_num] * neuron.derivative(this->neurons_inputs[layer_num][neuron_num], &(neuron.beta)));
        }
        else
        {
            double ret = 0;
            auto neuron = this->layers[layer_num].neurons[neuron_num];
            for (uint32_t i{0}; i < this->layers[layer_num + 1].neurons.size(); i++)
                ret += __get_delta(layer_num + 1, i, layer_num, neuron_num);

            //multiply by weight
            if (layer_num != origin_layer)
                ret *= this->layers[layer_num].neurons[neuron_num].weights[origin_neuron];

            ret *= neuron.derivative(this->neurons_inputs[layer_num][neuron_num], &(neuron.beta));
            return ret;
        }
    }

public:
    double learning_rate;
    double momentum;
    uint32_t prev_weights;

    std::vector<Layer> layers;

    data_set input;
    data_set target;

    /**
     * Train network
     */
    inline void train(std::vector<uint32_t> input_rows, data_row &costs)
    {
        for (uint32_t i{0}; i < this->batch_size; i++)
        {
            //Get cost for each input value
            costs[i] = this->get_cost(input_rows[i]);

            //First get deltas
            //then calculate weights and biases and put them in array

            for (uint32_t layers{0}; layers < this->layers_count; layers++)
            {
                for (uint32_t neurons{0}; neurons < this->layers[layers].neuron_count; neurons++){
                    Neuron &neuron = this->layers[layers].neurons[neurons];
                    double delta = this->get_delta(layers, neurons);
                    neuron.batch_bias[i] = neuron.bias + this->layers[layers].learning_rate * delta;
                    
                    for(uint32_t weights{0};weights < neuron.weights_count;weights++){
                        double out = 0;
                        if(layers == 0)
                            out = this->input[i][weights];
                        else
                            out = this->output[layers-1][weights];
                        neuron.batch_weights[i][weights] = neuron.weights[weights] + out * this->layers[layers].learning_rate * delta;
                    }
                }
            }
        }
    }

    /**
     * Set size of batch
     */
    void set_batch_size(uint32_t size);

    /**
     *  Get delta for given neuron
     */
    inline double get_delta(uint32_t layer_num, uint32_t neuron_num)
    {
        return __get_delta(layer_num, neuron_num, layer_num, neuron_num);
    }

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
        this->layers[0].feed(input, this->output[0], this->neurons_inputs[0]);

        for (uint32_t i{1}; i < layers.size(); i++)
            this->layers[i].feed(this->output[i - 1], this->output[i], this->neurons_inputs[i]);
    }

    /**
     * Basing on cost vector, fits network
     */
    inline void fit()
    {
        for (auto &layer : this->layers)
        {
            for (auto &neuron : layer.neurons)
                neuron.fit();
        }
    }

    /**
     * Add layer to net
     */
    void add_layer(uint32_t neurons, double learning_rate, double momentum, size_t prev_weights);

    /**
     * Add layer to net
     */
    void add_layer(uint32_t neurons);

    /**
     *  Constructor of neural network
     *  @arg input - input data set
     *  @arg target - target data set
     */
    Net(data_set &input, data_set &target, double learning_rate = 0.9, double momentum = 0.1, size_t prev_weights = 1);

    ~Net();
};
