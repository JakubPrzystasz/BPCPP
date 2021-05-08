#pragma once

#include "layer.h"

class Net
{
public:
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
     * SSE 
     */
    double SSE;

    /**
     * MSE
     */
    double MSE;

    /**
     * Train network
     */
    void train(uint32_t data_row_num);

    /**
     *  Constructor of neural network
     *  @arg input - input data set
     *  @arg target - target data set
     *  @arg layers - hidden layers vector
     */
    Net(data_set &input, data_set &target, std::vector<uint32_t> &layers, uint32_t batch_size = 1, double learning_rate = 0.001, double momentum_const = 0.1);

    ~Net();

    void feed(uint32_t data_row_num);
};

/*
TODO:
    Input data preparation - make internal vector of indexes of input data set
    every call of function traingdx suffle 3 sets of input data:
    train set, validate set, and test set. in default proportion 80,15,5
    but every set must have each class

    Make batch learning.

    Add adaptive learning rate, and momentum with variable momentum delta

*/