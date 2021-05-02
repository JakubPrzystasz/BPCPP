#include "neuron.h"

namespace ActivationFunction
{
    double unipolar(double input, double *params)
    {
        return 1.0 / (1.0 + pow(M_E, -((*params) * input)));
    }

    double unipolar_derivative(double base_output, double *params)
    {
        return (*params) * base_output * (1.0 - base_output);
    }

    double bipolar(double input, double *params)
    {
        return 2.0 / (1.0 + pow(M_E, -1 * ((*params) * input)) - 1.0);
    }

    double bipolar_derivative(double base_output, double *params)
    {
        return (*params) * (1.0 - pow(base_output, 2.0));
    }
};


Neuron::Neuron(uint32_t weights_count, uint32_t momentum_count, double bias, double beta, func_ptr activation, func_ptr activation_derivative)
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


Neuron::~Neuron() {}