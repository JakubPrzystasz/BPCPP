#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>
#include <cmath>
#include <limits>
#include <cstring>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>
#include <map>

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


/**
 * Returns random value in given range
 * @arg range - pair of double - first is min, second is max 
 */
double random_value(rand_range range);

/**
 * Activation functions and its derivatives of non-linear block of neuron
 */
namespace ActivationFunction
{
	inline double __normalize(double value)
	{
		if (std::isinf(value))
		{
			if (value > 0.0)
				value = std::numeric_limits<double>::max();
			else
				value = std::numeric_limits<double>::min();
		}
		return value;
	}

	/**
	 * Unipolar neuron activation function - as parameter it takes array of one element (Beta constant)
	 */
	double unipolar(double input, double *params);

	/**
	 * Unipolar neuron activation derivative function - as parameter it takes array of one element (Beta constant), and base function output for the same argument
	 */
	double unipolar_derivative(double input, double *params);

	/**
	 * Bipolar neuron activation function - as parameter it takes array of one element (Beta constant)
	 */
	double bipolar(double input, double *params);

	/**
	 * Bipolar neuron activation derivative function - as parameter it takes array of one element (Beta constant), and base function output for the same argument
	 */
	double bipolar_derivative(double input, double *params);

	/**
	 * Linear neuron activation 
	 */
	double purelin(double input, double *params);

	/**
	 * Linear neuron activation derivative
	 */
	double purelin_derivative(double input, double *params);
};


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
     * Pointer to activation function
     */
    func_ptr activation;

    /**
     * Pointer to derivative of activation function
     */
    func_ptr derivative;

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

        this->activation = ActivationFunction::bipolar;
        this->derivative = ActivationFunction::bipolar_derivative;
    };
};