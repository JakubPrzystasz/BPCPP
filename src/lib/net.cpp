#include "net.h"

void Net::add_layer(uint32_t neurons, double learning_rate, double momentum, size_t prev_weights)
{
    if (this->layers.size() < 1)
        this->layers.push_back(Layer(neurons, neurons, prev_weights, learning_rate, momentum));
    else
        this->layers.push_back(Layer(neurons, this->layers.back().neuron_count, prev_weights, learning_rate, momentum));

    //Setup helper containers
    this->cost = data_row(this->layers.back().neuron_count);
    this->output.erase(this->output.begin(), this->output.end());
    for (auto &layer : layers)
        this->output.push_back(data_row(layer.neuron_count));

    this->layers_count += 1;
}


Net::Net(data_set &input, data_set &target)
{
    this->input = input;
    this->target = target;
    this->cost = data_row(target[0].size(), 0);
}


Net::~Net() {}