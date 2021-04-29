#ifndef NEURON_H
#define NEURON_H
#include "includes.h"

/**
 * Unipolar neuron activation function - as parameter it takes array of one element (Beta constant)
 */
double unipolar_func(double input, double *params);

/**
 * Unipolar neuron activation derivative function - as parameter it takes array of one element (Beta constant), and base function output for the same argument
 */
double unipolar_func_derivative(double *base_output, double *params);

/**
 * Bipolar neuron activation function - as parameter it takes array of one element (Beta constant)
 */
double bipolar_func(double input, double *params);

/**
 * Bipolar neuron activation derivative function - as parameter it takes array of one element (Beta constant), and base function output for the same argument
 */
double bipolar_func_derivative(double *base_output, double *params);


/**
 * Implementation of simple neuron,
 * as template argument it takes number of inputs
 */
template <size_t input_count>
class Neuron
{

public:
    /**
     * Beta constans
     */
    double beta;

    /**
     * Bias of neuron
     */
    double bias;

    /**
     * Vector of input weights
     */
    std::array<double, input_count> input_weights;

    /**
     * Pointer to activation function 
     */
    static double (*acitivation_function)(double input, double *params);

    /**
     * Pointer to activation function derivative
     */
    static double (*acitivation_function_derivative)(double *base_output, double *params);

    /**
     * Get output value of neuron from given input value
     */
    double output(std::array<double, input_count> &input);

    /**
     * Class constructor, as parameter it takes argument for activation function 
     */
    Neuron(double* params);

    /**
     * Class destructor 
     */
    ~Neuron();
};

#endif