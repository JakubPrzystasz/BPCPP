#include "neuron.h"

namespace ActivationFunction
{   
    double __normalize(double value)
    {
        if (std::isinf(value))
        {
            if (value > 0.0)
                value = std::numeric_limits<double>::max();
            else
                value = std::numeric_limits<double>::min();
        }
        return value;
    }

    double unipolar(double input, double *params)
    {
        double ret = __normalize((1.0 / (1.0 + std::pow(M_E, -((*params) * input)))));
        if (ret > 1.0)
            return 1.0;
        if (ret < 0)
            return 0;
        return ret;
    }

    double unipolar_derivative(double input, double *params)
    {
        return __normalize((*params) * input * (1.0 - input));
    }

    double bipolar(double input, double *params)
    {
        double ret = __normalize(tanh((*params) * input));
        if (ret > 1.0)
            return 1.0;
        if (ret < -1.0)
            return -1.0;
        return ret;
    }

    double bipolar_derivative(double input, double *params)
    {
        return __normalize((*params) * (1.0 - std::pow(input, 2.0)));
    }

    double purelin(double input, __attribute__ ((unused)) double *params)
    {
        return input;
    }

    double purelin_derivative(__attribute__ ((unused))double input, __attribute__ ((unused))double *params)
    {
        return 1.0;
    }
}

Neuron::Neuron(uint32_t inputs, LearnParams params) : batch(inputs, params.batch_size)
{
    if (!params.batch_size)
        throw std::invalid_argument(std::string("Batch size can not be 0"));

    this->learn_parameters = params;

    // Assign pointers to activation functions
    this->activation = this->learn_parameters.activation;
    this->derivative = this->learn_parameters.derivative;

    //Initialize weights and bias with random values
    this->weights = data_row(inputs, 0);

    this->bias = 0;

    //Weight delta for momentum method
    this->weights_deltas = data_set(inputs, data_row(this->learn_parameters.momentum_delta_vsize, 0));
    this->bias_deltas = data_row(this->learn_parameters.momentum_delta_vsize, 0);

    this->weight_update = data_row(inputs, 0);
}

Neuron::~Neuron() {}
