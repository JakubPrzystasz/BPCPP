#pragma once

#include "layer.h"
#include <cmath>

class Net
{
    /**
     * Cost vector
     */
    data_row cost;

    /**
     * Size of batch
     */
    uint32_t batch_size;

    /**
     * Size of input layer
     */
    uint32_t input_size;

    /**
     * Size of output layer
     */
    uint32_t output_size;
public:

    /**
     * Input data set
     */
    data_set input;

    /**
     * Target data set
     */
    data_set target;

    /**
     * Vector of layers
     */
    std::vector<Layer> layers;

    /**
     * Learning rate
     */
    double learning_rate;

    /**
     * Momentum constans
     */
    double momentum_const;

    /**
     * Train network
     */
    // inline void train(std::vector<uint32_t> input_rows, data_row &costs)
    // {
    //     for (uint32_t i{0}; i < this->batch_size; i++)
    //     {
    //         //Get cost for each input value
    //         costs[i] = this->get_cost(input_rows[i]);

    //         //delta
    //         for (uint32_t layer{this->layers_count - 1}; layer > 0; layer--)
    //         {
    //             for (uint32_t neuron{0}; neuron < this->layers[layer].neuron_count; neuron++)
    //             {
    //                 Neuron *_neuron = &(this->layers[layer].neurons[neuron]);

    //                 if (layer == (this->layers_count - 1))
    //                     _neuron->delta = this->cost[neuron] * _neuron->derivative(this->neurons_inputs[layer][neuron], &_neuron->beta);
    //                 else
    //                 {
    //                     double sum = 0;
    //                     for (uint32_t next{0}; next < this->layers[layer + 1].neuron_count; next++)
    //                     {
    //                         auto &next_neuron = this->layers[layer + 1].neurons[next];
    //                         sum += next_neuron.weights[neuron] * next_neuron.delta;
    //                     }
    //                     //calculate the delta for current neuron
    //                     double x = sum * _neuron->derivative(this->neurons_inputs[layer][neuron], &_neuron->beta);

    //                     _neuron->delta = x;
    //                 }
    //             }
    //         }

    //         for (uint32_t layer{1}; layer < this->layers_count; layer++)
    //         {
    //             for (uint32_t neuron{0}; neuron < this->layers[layer].neuron_count; neuron++)
    //             {
    //                 Neuron *_neuron = &(this->layers[layer].neurons[neuron]);

    //                 //then calculate weights and biases and put them in array
    //                 _neuron->batch_bias[i] = _neuron->bias - (this->layers[layer].learning_rate * _neuron->delta);

    //                 for (uint32_t weights{0}; weights < _neuron->weights_count; weights++)
    //                 {
    //                     double out = 0;
    //                     if (layer == 0)
    //                         out = this->input[input_rows[i]][weights];
    //                     else
    //                         out = this->output[layer - 1][weights];
    //                     _neuron->batch_weights[i][weights] = _neuron->weights[weights] - (out * this->layers[layer].learning_rate * _neuron->delta);
    //                 }
    //             }
    //         }
    //     }
    // }

    /**
     *  Constructor of neural network
     *  @arg input - input data set
     *  @arg target - target data set
     *  @arg layers - hidden layers vector
     */
    Net(data_set &input, data_set &target, std::vector<uint32_t> &layers, uint32_t batch_size = 1, double learning_rate = 0.1, double momentum_const = 0.1);

    ~Net();
};
