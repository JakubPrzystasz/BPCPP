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
    this->layers.push_back(Layer(this->input_size, 0, learning_rate, momentum_const, 0, (Layer *)nullptr));

    for (uint32_t i{0}; i < layers.size(); i++)
        this->layers.push_back(Layer(layers[i], (i > 0 ? layers[i - 1] : input_size), learning_rate, momentum_const, i, (Layer *)&layers.back()));

    //output layer
    this->layers.push_back(Layer(this->output_size, this->layers.back().neurons.size(), learning_rate, momentum_const, layers.size(), (Layer *)&layers.back()));

    //Setup cost vector
    this->cost = data_row(output_size, 0);
}

Net::~Net() {}
