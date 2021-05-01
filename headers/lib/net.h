#ifndef NET_H
#define NET_H

#include "layer.h"
#include "includes.h"

typedef std::vector<double> data_row;
typedef std::vector<data_row> data_set;

class Net
{
public:
    std::vector<Layer> layers;
    
    std::array<std::vector<double>,2> IO; 

    uint32_t get_max_neurons(){
        uint32_t ret = 0;
        for(auto &layer: layers)
            ret = layer.neurons.size() > ret ? layer.neurons.size() : ret; 
        return ret;
    }

    double train();

    /**
     * Feed net, 
     */
    void feed(std::vector<double> &inputs, std::vector<double> &outputs){
        uint8_t io_ptr = 0; 
        IO[1] = inputs;

        for(auto &layer: layers){
            io_ptr += 1;
            io_ptr %= 2;
            layer.feed(IO[io_ptr],IO[!io_ptr]);
        }
        outputs = IO[!io_ptr];
    }

    void add_layer(uint32_t neurons, double learning_rate = 0.1, double momentum = 0.1, size_t prev_weights = 1){
        if(layers.size() < 1){
            layers.push_back(Layer(neurons,neurons,prev_weights,learning_rate,momentum));
            return;
        }

        layers.push_back(Layer(neurons,layers.back().neuron_count,prev_weights,learning_rate,momentum));
        
        IO[0] = std::vector<double>(get_max_neurons(),0);
        IO[1] = std::vector<double>(get_max_neurons(),0);
    }

    /**
     *  Constructor of neural network
     */
    Net(data_set &input, data_set &target){
        input.size();
        target.size();
    }

    ~Net(){}
};

#endif