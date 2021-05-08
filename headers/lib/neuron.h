#pragma once

#include "includes.h"

namespace ActivationFunction
{	
	inline double __normalize(double value){
		if(std::isinf(value))
        {
            if(value > 0.0)
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

	double purelin(double input, double *params);
	double purelin_derivative(double input, double *params);
};

/**
 * Implementation of simple neuron,
 */
class Neuron
{
public:
	
	/**
	 * Output of activation function
	 */
	double output;
	
	/**
	 * Input value to activation function
	 */
	double input;
	
	/**
	 * Output of derivative function
	 */
	double derivative_output;

	/**
	 * Delta value
	 */
	double delta;


	/**
     * Bias of neuron
     */
	double bias;

	/**
	 * Vector of weights
	 */
	data_row weights;

	/**
     * Pointer to activation function 
     */
	func_ptr activation;

	/**
     * Pointer to activation function derivative
     */
	func_ptr derivative;

	/**
	 * Beta parameter
	 */
	double beta_param;

	/**
     * @arg inputs - number of neurons in previous layer
     */
	Neuron(uint32_t inputs, double rand_min, double rand_max);

	~Neuron();
};
