#pragma once

#include "includes.h"

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
     * Vector of input weights, and previous weights
     */
  std::vector<double> weights;

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
  inline double feed(std::vector<double> &input)
  {
    double output = bias;
    for (size_t i{0}; i < weights_count; i++)
      output += input[i] * weights[i];

    return (this->base(output, &(this->beta)));
  }

  /**
     * Neuron construction, as required arg takes vector of weights
     */
  Neuron(uint32_t weights_count, uint32_t momentum_count = 1, double bias = 0.0, double beta = 1.0, func_ptr activation = ActivationFunction::unipolar, func_ptr activation_derivative = ActivationFunction::unipolar_derivative);

  /**
     * Class destructor 
     */
  ~Neuron();
};
