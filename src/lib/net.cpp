#include "net.h"

Net::Net(data_set &input, data_set &target, std::vector<uint32_t> &layers, uint32_t batch_size, double learning_rate, double momentum_const)
{
    this->learning_rate = learning_rate;
    this->momentum_const = momentum_const;

    this->batch_size = batch_size;

    this->input = input;
    this->target = target;

    this->input_size = input.front().size();
    this->output_size = target.front().size();

    //Setup layers
    this->layers = std::vector<Layer>();
    //input layer
    this->layers.push_back(Layer(this->input_size, this->input_size, learning_rate, momentum_const, -1.0, 1.0));

    for (uint32_t i{0}; i < layers.size(); i++)
        this->layers.push_back(Layer(layers[i], (i > 0 ? layers[i - 1] : input_size), learning_rate, momentum_const, -1.0, 1.0));

    //output layer
    this->layers.push_back(Layer(this->output_size, this->layers.back().neurons.size(), learning_rate, momentum_const, -0.5, 0.5));

    //Setup cost vector
    this->cost = data_row(output_size, 0);
}

Net::~Net() {}

void Net::feed(uint32_t data_row_num)
{

    //First set input layer

    auto &layer = this->layers[0];
    auto &data = this->input[data_row_num];

    for (uint32_t neuron_it{0}; neuron_it < layer.neurons.size(); neuron_it++)
    {
        auto &neuron = layer.neurons[neuron_it];

        neuron.input = neuron.output = neuron.derivative_output = data[neuron_it];
    }

    //Set hidden layers and output layer
    for (uint32_t layer_it{1}; layer_it < this->layers.size(); layer_it++)
    {
        auto &layer = this->layers[layer_it];
        auto &prev_layer = this->layers[layer_it - 1];

        for (auto &neuron : layer.neurons)
        {
            neuron.output = neuron.bias;
            for (uint32_t weight_it{0}; weight_it < neuron.weights.size(); weight_it++)
                neuron.output += (neuron.weights[weight_it] * prev_layer.neurons[weight_it].output);
            neuron.output = neuron.activation(neuron.output, &(neuron.beta_param));
            neuron.derivative_output = neuron.derivative(neuron.output, &(neuron.beta_param));
        }
    }
}
