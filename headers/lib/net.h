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
     * Computes delta for each layer
     * @arg index at input_data vector
     */
    void get_delta(uint32_t sample_number);

public:

    /**
     * Read data form text file
     * put all data to input_data vector
     * @arg filename - name of file with input data
     * @arg input_data - reference to container for read data
     */
    static void read_file(std::string filename, pattern_set &input_data);

    /**
	 * Container for learning params, for all neurons in layer
     * it is kind of blue print for making new neurons in that layer
	 */
    LearnParams learn_parameters;

    /**
     * Percentage representation of training, validation, and test set
     * Sum of values in array must be equal 1
     */
    std::array<double,3> subsets_ratio;

    /**
     * Sum squared error - value computed and the end of each epoch
     */
    double SSE;

    /**
     * Mean square error - mean of SSE/size of set
     */
    double MSE;
    
    /**
     * Contains input, hidden, and output layers
     */
    std::vector<Layer> layers;


    /**
     * Trains network
     * @arg max_epoch - maximum number of training epochs
     * @arg error_goal - when SSE is less or equal error goal, stops training 
     */
    void train(double max_epoch = 1000, double error_goal = 0);

    /**
     * Feed network with sample
     * just gets output value of each neuron
     * @arg sample_number - index of sample in input_data vector
    */
    void feed(uint32_t sample_number);

    /**
     * Calculate delta for each neuron
     * @arg sample_number - index of sample in input_data vector
     */
    void get_delta(uint32_t sample_number)

    /** 
     * Setup internal vectors according to net properties
     * @arg hidden_layers - vector of sizes of hidden layers
     * @arg params - blueprint for each neuron 
     */
    void setup(std::vector<uint32_t> hidden_layers, LearnParams params);

    /**
     *  Constructor of neural network
     *  @arg input_data - input data set
     *  Given data set will be divided to subsets: training, validation and test set
     *  Each call of train makes new sets
     *  @arg subsets_ratio - Percentage representation of training, validation, and test set
     */
    Net(pattern_set &input_data, std::array<double,3> subsets_ratio);

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