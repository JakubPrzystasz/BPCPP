#pragma once

#include "layer.h"

class Net
{
    enum class LearningRateUpdate
    {
        Increase,
        Decrease,
    };

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
     * Subsets
     */
    std::vector<uint32_t> train_set;
    std::vector<uint32_t> test_set;

    /**
     *
     */
    std::vector<uint32_t> indexes;

    uint32_t batch_it;

    double prev_SSE;

    uint32_t epoch;

    std::vector<std::pair<double, std::vector<uint32_t>>> class_sets;

    std::vector<std::pair<double, std::vector<uint32_t>>>::iterator get_class_vector(double class_value);

    void update_learning_rate(LearningRateUpdate value);

public:
    /**
     * Defines saving mode
     */
    enum class SaveMode
    {
        Overwrite,
        Append,
    };

    /**
     * Read data form text file
     * put all data to input_data vector
     * @arg filename - name of file with input data
     * @arg input_data - reference to container for read data
     */
    static void read_file(std::string filename, pattern_set &input_data);

    /**
     * Save learn output to desired file
     */
    static void save_output(std::string filename, LearnOutput &output, SaveMode mode = SaveMode::Append);

    static void open_file(std::string filename);
    static void close_file(std::string filename);

    /**
	 * Container for learning params, for all neurons in layer
     * it is kind of blue print for making new neurons in that layer
	 */
    LearnParams learn_parameters;

    /**
     * Percentage representation of training, and test set
     * Sum of values in array must be equal 1
     */
    std::array<double, 2> subsets_ratio;

    /**
     * Sum squared error - value computed and the end of each epoch
     */
    double SSE;

    /**
     * Contains input, hidden, and output layers
     */
    std::vector<Layer> layers;

    /**
     * Trains network
     * @arg max_epoch - maximum number of training epochs
     * @arg error_goal - when SSE is less or equal error goal, stops training 
     * @arg return - LearningOutput
     */
    LearnOutput train(double max_epoch = 1000, double error_goal = 0);

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
    void get_delta(uint32_t sample_number);

    /**
     * Calculates error
     * @arg sample_number - number of sample in input_data vector
     */
    double get_cost(uint32_t sample_number);

    /**
     * Updates weights for all neurons in net
     */
    void update_weights();

    /**
     * Calculate gradient
     * @arg sample_number - number of sample in input_data vector
     */
    void learn(uint32_t sample_number);

    /**
     * If classification succeeds, returns true otherwise false
     * @arg sample_number - number of sample in input_data vector
     */
    bool get_classification_succes(uint32_t sample_number);

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
    Net(pattern_set &input_data, std::array<double, 2> subsets_ratio);

    ~Net();
};
