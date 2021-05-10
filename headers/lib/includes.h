#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <limits>
#include <cstring>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>

struct Neuron;
struct Layer;
struct Pattern;
struct LearnParams;
class Net;
typedef std::vector<double> data_row;
typedef std::vector<data_row> data_set;
typedef std::vector<Pattern> pattern_set;
typedef std::pair<double, double> rand_range;
typedef double (*func_ptr)(double, double *);

struct Pattern
{
    /**
     * Input vector
     */
    data_row input;

    /**
     *Output vector
     */
    data_row output;
};

/**
    *  Learning parameters of neuron/layer/network
*/
struct LearnParams
{
    /**
		 * Batch size
		 */
    uint32_t batch_size;

    /**
        * Initial learning rate
    	*/
    double learning_rate;

    /**
        * Error ratio
     	*/
    double error_ratio;

    /**
        * Aka. gamma+ 
		* Note: to use adaptive learning both values must be set
     	*/
    double learning_accelerating_constans;

    /**
        * Aka. gamma- 
		* Note: to use adaptive learning both values must be set
     	*/
    double learning_decelerating_constans;

    /**
        * Momentum constans
    	*/
    double momentum_constans;

    /**
		 * Weights deltas vector size
		 */
    uint32_t momentum_delta_vsize;

    /**
		 * Beta parameter used in activation function
		 */
    double beta_param;

    /**
		 * Initial range for weights
		 */
    rand_range weights_range;

    /**
		 * Initial range for bias
		 */
    rand_range bias_range;

    /**
		 * Default constructor
		 * sets all values to default
		 */
    LearnParams()
    {
        //Default learn method is stochastic
        this->batch_size = 1;

        this->learning_rate = 0.1;

        this->learning_accelerating_constans = 1.01;
        this->learning_decelerating_constans = 0.99;

        this->momentum_constans = 0.95;

        //Size of vector for weights deltas for each neuron
        //Vector of deltas is necessary for momentum method
        this->momentum_delta_vsize = 1;

        this->beta_param = 1.0;

        this->weights_range = std::make_pair(-1.0, 1.0);
        this->bias_range = std::make_pair(-1.0, 1.0);
    };
};

double random_value(rand_range &range);