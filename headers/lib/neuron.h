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
    double feed(std::vector<double> &input)
    {
        double output = bias;
        for (size_t i{0}; i < weights_count; i++)
            output += input[i] * weights[i];

        return (this->base(output, &(this->beta)));
    }

    /**
     * Neuron construction, as required arg takes vector of weights
     */
    Neuron(uint32_t weights_count, uint32_t momentum_count = 1, double bias = 1.0, double beta = 1.0, func_ptr activation = ActivationFunction::unipolar, func_ptr activation_derivative = ActivationFunction::unipolar_derivative)
    {
        this->bias = bias;
        this->beta = beta;
        this->base = activation;
        this->derivative = activation_derivative;
        this->weights_count = weights_count;
        this->momentum_count = momentum_count;

        this->weights = std::vector<double>(weights_count * (1 + momentum_count), 0);

        std::random_device r;
        std::default_random_engine gen(r());
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        for (double &weight : this->weights)
            weight = dis(gen);
    }

    /**
     * Class destructor 
     */
    ~Neuron()
    {
    }
};

#endif