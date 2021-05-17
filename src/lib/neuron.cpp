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
        double ret = __normalize((1.0 / (1.0 + pow(M_E, -((*params) * input)))));
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
        return __normalize((*params) * (1.0 - pow(input, 2.0)));
    }

    double purelin(double input, double *params)
    {
        return input;
    }

    double purelin_derivative(double input, double *params)
    {
        return 1.0;
    }
};

namespace InitFunction
{
    double random_range(void *params)
    {
        rand_range &range = *(rand_range *)params;
        std::mt19937_64 rng;
        uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32)};
        rng.seed(ss);
        std::uniform_real_distribution<double> unif(static_cast<double>(std::get<0>(range)), static_cast<double>(std::get<1>(range)));
        return unif(rng);
    }

};

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

    for (auto &weight : this->weights)
        weight = this->learn_parameters.init_function((void *)&this->learn_parameters.weights_range);

    this->bias = this->learn_parameters.init_function((void *)&this->learn_parameters.bias_range);

    //Weight delta for momentum method
    this->weights_deltas = data_set(inputs, data_row(this->learn_parameters.momentum_delta_vsize, 0));
    this->bias_deltas = data_row(this->learn_parameters.momentum_delta_vsize, 0);

    this->weight_update = data_row(inputs, 0);
}

Neuron::~Neuron() {}
