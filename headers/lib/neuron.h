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
      this->bias = 0;
      for (auto &value : this->batch_bias)
         this->bias += value;

      this->bias = this->bias / batch_size;

      //move last used weights to next array:
      for (uint32_t x{1}; x <= this->momentum_count; x++)
      {
         for (uint32_t i{0}; i < this->weights_count; i++)
            this->weights[i + (x * this->weights_count)] = this->weights[i + ((x - 1) * this->weights_count)];
      }

      for (uint32_t i{0}; i < this->weights_count; i++)
         this->weights[i] = 0;

      //calculate new weights according to mean of weights in batch
      for (auto &row : this->batch_weights)
      {
         for (uint32_t i{0}; i < weights_count; i++)
            this->weights[i] += row[i];
      }

      for (uint32_t i{0}; i < this->weights_count; i++){
         this->weights[i] = this->weights[i] / this->batch_size;
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

      return this->base(input_value, &(this->beta));
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
