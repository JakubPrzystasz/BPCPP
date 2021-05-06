#pragma once

#include "includes.h"
#include <cmath>

namespace ActivationFunction
{
	/**
 * Unipolar neuron activation function - as parameter it takes array of one element (Beta constant)
 */
	double unipolar(double input, double *params);

	/**
 * Unipolar neuron activation derivative function - as parameter it takes array of one element (Beta constant), and base function output for the same argument
 */
	double unipolar_derivative(double base_output, double *params);

	/**
 * Bipolar neuron activation function - as parameter it takes array of one element (Beta constant)
 */
	double bipolar(double input, double *params);

	/**
 * Bipolar neuron activation derivative function - as parameter it takes array of one element (Beta constant), and base function output for the same argument
 */
	double bipolar_derivative(double base_output, double *params);
};

/**
 * Implementation of simple neuron,
 */
class Neuron
{
public:
	data_set batch_weights;
	data_row batch_bias;
	uint32_t batch_size;

	double delta;

	inline void fit()
	{
		this->bias = this->batch_bias[0];

        for(uint32_t i{0}; i < this->weights_count;i++){
            this->weights[i] = this->batch_weights[0][i];
        }
	}

	/**
     * Number of weights
     */
	uint32_t weights_count;

	/**
     * Number of weights
     */
	uint32_t momentum_count;

	/**
     * Bias of neuron
     */
	double bias;

	/**
     * Beta parameter
     */
	double beta;

	/**
     * Vector of input weights, and previous weights(momentum)
     */
	data_row weights;

	/**
     * Pointer to activation function 
     */
	func_ptr base;

	/**
     * Pointer to activation function derivative
     */
	func_ptr derivative;

	/**
     * Get output value of neuron from given input value
     */
	inline double feed(data_row &inputs, double &input_value)
	{
		input_value = bias;
		for (uint32_t i{0}; i < weights_count; i++)
			input_value += inputs[i] * weights[i];

		double ret = this->base(input_value, &(this->beta));

		if(std::isnan(ret) || std::isinf(ret))
		    exit(10);

		return ret;
	}

	/**
     * Neuron construction, as required arg takes vector of weights
     */
	Neuron(uint32_t weights_count, uint32_t momentum_count = 1, double beta = 1.0, func_ptr activation = ActivationFunction::bipolar, func_ptr activation_derivative = ActivationFunction::bipolar_derivative);

	/**
     * Class destructor 
     */
	~Neuron();
};
