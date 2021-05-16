#pragma once

#include "includes.h"
#include <numeric>

/**
 * Implementation of simple neuron,
 */
struct Neuron
{
	/**
	 * Container for learning params
	 */
	LearnParams learn_parameters;

	/**
	 * Batch updates values for each input
	 */
	Batch batch;

	data_set weights_deltas;
	data_row bias_deltas;

	data_row weight_update;
	double bias_update;

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

	double gradient;

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
     * @arg inputs - number of neurons in previous layer
	 * @arg params - neuron learning parameters 
     */
	Neuron(uint32_t inputs, LearnParams params);

	~Neuron();
};
