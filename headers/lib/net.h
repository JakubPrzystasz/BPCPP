#pragma once

#include "layer.h"
#include <cmath>

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

    void __add_layer(uint32_t neurons, double learning_rate, double momentum, uint32_t prev_weights);

    inline double __get_delta(uint32_t layer_num, uint32_t neuron_num, uint32_t origin_layer, uint32_t origin_neuron)
    {
        if (layer_num == this->layers_count - 1)
        {
            auto neuron = this->layers[layer_num].neurons[neuron_num];
            double weight = 1.0;
            if (layer_num != origin_layer)
                weight = neuron.weights[origin_neuron];
            double ret = (weight * this->cost[neuron_num] * neuron.derivative(this->neurons_inputs[layer_num][neuron_num], &(neuron.beta)));
            return ret;
        }
        else
        {
            double ret = 0;
            auto neuron = this->layers[layer_num].neurons[neuron_num];
            for (uint32_t i{0}; i < this->layers[layer_num + static_cast<uint32_t>(1)].neurons.size(); i++)
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

            //delta
            for (uint32_t layer{this->layers_count - 2};; layer--)
            {
                for (uint32_t neuron{0}; neuron < this->layers[layer].neuron_count; neuron++)
                {
                    auto &_neuron = this->layers[layer].neurons[neuron];

                    if (layer == this->layers_count - 1)
                        _neuron.delta = this->cost[neuron] * _neuron.derivative(this->neurons_inputs[layer][neuron], &_neuron.beta);
                    else
                    {
                        double sum = 0;
                        for (uint32_t next{0}; next < this->layers[layer+1].neuron_count; next++)
                        {
                            auto &next_neuron = this->layers[layer + 1].neurons[next];
                            sum += next_neuron.weights[neuron] * next_neuron.delta;
                        }
                        //calculate the delta for current neuron
                        _neuron.delta = sum * _neuron.derivative(this->neurons_inputs[layer][neuron], &_neuron.beta);
                    }

                    //then calculate weights and biases and put them in array
                    _neuron.batch_bias[i] = _neuron.bias - this->layers[layer].learning_rate * _neuron.delta;

                    for (uint32_t weights{0}; weights < _neuron.weights_count; weights++)
                    {
                        double out = 0;
                        if (layer == 0)
                            out = this->input[i][weights];
                        else
                            out = this->output[layer - 1][weights];
                        _neuron.batch_weights[i][weights] = _neuron.weights[weights] - out * this->layers[layer].learning_rate * _neuron.delta;
                    }
                }

                if (!layer)
                    break;
            }
        }
    }

    /**
     * Set size of batch
     */
    void
    set_batch_size(uint32_t size);

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
    void add_layer(uint32_t neurons, double learning_rate, double momentum, uint32_t prev_weights);

    /**
     * Add layer to net
     */
    void add_layer(uint32_t neurons);

    /**
     *  Constructor of neural network
     *  @arg input - input data set
     *  @arg target - target data set
     */
    Net(data_set &input, data_set &target, double learning_rate = 0.001, double momentum = 0.1, uint32_t prev_weights = 1);

    ~Net();
};
