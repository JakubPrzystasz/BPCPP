#include "neuron.h"

double random_value(double min, double max)
{
    std::mt19937_64 rng;
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32)};
    rng.seed(ss);
    std::uniform_real_distribution<double> unif(min, max);
    return unif(rng);
}

namespace ActivationFunction
{
    double unipolar(double input, double *params)
    {
        return __normalize((1.0 / (1.0 + pow(M_E, -((*params) * input)))));
    }

    double unipolar_derivative(double input, double *params)
    {
        return __normalize((*params) * input * (1.0 - input));
    }

    double bipolar(double input, double *params)
    {
        double ret = __normalize(tanh((*params) * input));
        if (ret < -1.0)
            return -1.0;
        if (ret > 1.0)
            return 1.0;
        return ret;
    }

    double bipolar_derivative(double input, double *params)
    {
        return __normalize((*params) * (1.0 - pow(input, 2.0)));
    }

    double purelin(double input, double *params)
    {
        if (input > 1.0)
            return 1.0;
        if (input < -1.0)
            return -1.0;
        return input;
    }

    double purelin_derivative(double input, double *params)
    {
        return 1.0;
    }
};

Neuron::Neuron(uint32_t inputs, double rand_min, double rand_max, func_ptr activation, func_ptr derivative)
{

    this->activation = activation;
    this->derivative = derivative;

    this->weights = data_row(inputs);

    for (auto &weight : this->weights)
        weight = random_value(rand_min, rand_max);

    this->bias = random_value(rand_min, rand_max);

    this->beta_param = 1.0;
}

Neuron::~Neuron() {}
