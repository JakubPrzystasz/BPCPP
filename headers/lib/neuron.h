#ifndef NEURON_H
#define NEURON_H
#include "includes.h"

typedef double (*func_ptr)(double, double *);

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
 * @arg n number of inputs
 * @arg momentum number of previous momentum m-1, m-2...
 */
template <size_t n, size_t m>
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
     * Vector of input weights, and previous weights
     */
    std::array<double, n * (m+1)> input_weights;

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
    double feed(std::array<double, n> &input)
    {
        double output = bias;
        for (size_t i{0}; i < n; i++)
            output += input[i] * input_weights[i];

        return (this->base(output, &(this->beta)));
    }

    /**
     * Neuron construction, as required arg takes vector of weights
     */
    Neuron(func_ptr activation = ActivationFunction::unipolar, func_ptr activation_derivative = ActivationFunction::unipolar_derivative, double beta = 1.0, double bias = 1.0)
    {
        this->bias = bias;
        this->beta = beta;
        this->base = activation;
        this->derivative = activation_derivative;

        std::random_device r;
        std::default_random_engine gen(r());
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        for (size_t i{0}; i < n; i++)
            input_weights[i] = dis(gen);
    }

    /**
     * Neuron construction, as required arg takes vector of weights
     * it makes weights random
     */
    Neuron(std::array<double, n> &weights, func_ptr activation = ActivationFunction::unipolar, func_ptr activation_derivative = ActivationFunction::unipolar_derivative, double beta = 1.0, double bias = 1.0)
    {
        this->bias = bias;
        this->beta = beta;
        this->base = activation;
        this->derivative = activation_derivative;
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