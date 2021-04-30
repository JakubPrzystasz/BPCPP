#ifndef NET_H
#define NET_H

#include "layer.h"
#include "includes.h"

class Net
{
public:
    std::vector<Layer> layers;


    /**
     * Call feed for all layers recursively
     */
    void feed(std::vector<double> &inputs, std::vector<double> &outputs){
        std::vector<double> _in = inputs;
        std::vector<double> _out;
        
        for(auto &layer: layers){
            _out = std::vector<double>(layer.neuron_count);
            layer.feed(_in,_out);
            _in = _out;
        }

        outputs = _out;
    }

    void add_layer(uint32_t neurons, double learning_rate = 0.1, double momentum = 0.1, size_t prev_weights = 1){
        if(layers.size() < 1){
            layers.push_back(Layer(neurons,neurons,prev_weights,learning_rate,momentum));
            return;
        }

        layers.push_back(Layer(neurons,layers.back().neuron_count,prev_weights,learning_rate,momentum));
    }

    /**
     *  Constructor of neural network
     */
    Net(){}

    ~Net(){}
};

#endif