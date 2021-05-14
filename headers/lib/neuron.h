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
	 * Updates weights computed in batch
	 */
	void update_weights()
	{	
		double tmp;
		for (uint32_t it{0}; it < this->weights.size(); it++){
			tmp = std::accumulate(batch.weights_deltas[it].begin(), batch.weights_deltas[it].end(), 0.0) / static_cast<double>(this->learn_parameters.batch_size); 
			this->weight_update[it] += tmp;
			this->weights[it] += tmp;
		}

		tmp = std::accumulate(this->batch.bias_deltas.begin(), this->batch.bias_deltas.end(), 0.0) / static_cast<double>(this->learn_parameters.batch_size);
		this->bias_update += tmp;
		this->bias += tmp; 
	}

	/**
     * @arg inputs - number of neurons in previous layer
	 * @arg params - neuron learning parameters 
     */
	Neuron(uint32_t inputs, LearnParams params);

	~Neuron();
};
