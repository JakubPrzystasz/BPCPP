#pragma once

#include "layer.h"

class Net
{
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
    pattern_set input_data;

    /**
     * Iterator over values in single batch
     */
    uint32_t batch_it;

    /**
     * Previous value of SSE - for adaptive learning rate
     */
    double SSE_previous;


    /**
     * Train network
     * computes SSE, MSE, outputs
     * It just does one Epoch of training set
     */
    void __train(uint32_t sample_number);
public:

    /**
     * Read data form text file
     * put all data to input_data vector
     */
    static void read_file(std::string filename, pattern_set &input_data);

    /**
         * Size of batch - it can be value in range 1 - number of samples
         * if 0 - full batch (adjusted to number of samples for train set, validation set, and test set)
         * if 1 - stochastic learning
         * if greater - mini batch, but can not exceed number of samples in set
         */
    uint32_t batch_size;

    /**
        * Learning rate must be less than 1  
    */
    double learning_rate;

    /**
        * Momentum constans
    */
    double momentum_constans;

    /**
        * Aka. gamma+ (page 160. Adaptive Learning of Polynominals)
     */
    double learning_accelerating_constans;

    /**
        * Aka. gamma+ (page 160. Adaptive Learning of Polynominals)
     */
    double learning_decelerating_constans;




    /**
     * Range for random values of initial weights and bias for output layer neurons
     */
    std::pair<double,double> output_layer_range;

    /**
     * SSE 
     */
    double SSE;

    /**
     * MSE
     */
    double MSE;

    
    /**
     * Vector of all layers
     */
    std::vector<Layer> layers;

    /**
     * Feed network with sample
     * just gets output value of each neuron
     */
    void feed(uint32_t sample_number);


    void train(double max_epoch = 1000, double error_goal = 0);


    /**
     *  Constructor of neural network
     *  @arg input - input data set
     *  @arg target - target data set
     *  @arg layers - hidden layers vector
     */
    Net(pattern_set &input_data);


    /**
     * Setup internal vectors according to class properties
     */
    void setup(std::vector<uint32_t> &hidden_layers, uint32_t batch_size = 1);

    ~Net();
};

/*
TODO:
    Input data preparation - make internal vector of indexes of input data set
    every call of function traingdx suffle 3 sets of input data:
    train set, validate set, and test set. in default proportion 80,15,5
    but every set must have each class

    Questions:
    can i use bigger momentum delta vector?
    is it ok to use just one method of adaping learning rate, or make experiment with it? 

*/