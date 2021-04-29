#ifndef NEURON_H
#define NEURON_H
#include "includes.h"

typedef double (*func_ptr)(double, double *);

namespace ActivationFunction
{
    /**
 * Unipolar neuron activation function - as parameter it takes array of one element (Beta constant)
 */
    double unipolar_func(double input, double *params);

    /**
 * Unipolar neuron activation derivative function - as parameter it takes array of one element (Beta constant), and base function output for the same argument
 */
    double unipolar_func_derivative(double base_output, double *params);

    /**
 * Bipolar neuron activation function - as parameter it takes array of one element (Beta constant)
 */
    double bipolar_func(double input, double *params);

    /**
 * Bipolar neuron activation derivative function - as parameter it takes array of one element (Beta constant), and base function output for the same argument
 */
    double bipolar_func_derivative(double base_output, double *params);
};

/**
 * Implementation of simple neuron,
 * as argument it takes number of inputs
 */
template <size_t n>
class Neuron
{
    double input_size = n;

public:
    /**
     * Bias of neuron
     */
    double bias;

    /**
     * Beta parameter
     */
    double beta;

    /**
     * Vector of input weights
     */
    std::array<double, n> input_weights;

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
    double output(std::array<double, n> &input)
    {
        double output = bias;
        for (size_t i{0}; i < n; i++)
            output += input[i] * input_weights[i];

        return (this->base(output, &(this->beta)));
    }

    /**
     * Neuron construction, as required arg takes vector of weights
     */
    Neuron(std::array<double, n> &weights, func_ptr activation_func = ActivationFunction::unipolar_func, func_ptr activation_derivative_func = ActivationFunction::unipolar_func_derivative, double beta = 1.0, double bias = 1.0)
    {
        this->bias = bias;
        this->beta = beta;
        this->base = activation_func;
        this->derivative = activation_derivative_func;
        input_weights = weights;
    }

    /**
     * Class destructor 
     */
    ~Neuron()
    {
    }
};

#endif