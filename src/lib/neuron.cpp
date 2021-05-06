#include "neuron.h"
#include <limits>

namespace ActivationFunction
{
    double unipolar(double input, double *params)
    {
        if(std::isinf(input))
        {
            if(input < 0)
                return 0.0;
            else
                return 1.0;
        }

        if(std::isnan(input))
            return 0.0;

        return (1.0 / (1.0 + pow(M_E, -((*params) * input))));
    }

    double unipolar_derivative(double base_output, double *params)
    {
        double ret =  ((*params) * base_output * (1.0 - base_output));
        if(std::isinf(ret)){
            if(ret > 0)
                return std::numeric_limits<double>::max();
            else
                return std::numeric_limits<double>::max();
        }

        return ret;
    }

    double bipolar(double input, double *params)
    {
        if(std::isinf(input))
        {
            if(input < 0)
                return -1.0;
            else
                return 1.0;
        }

        double x = sin(*params);
        x++;

        if(std::isnan(input))
            return 0.0;


        return tanh(input);
    }

    double bipolar_derivative(double base_output, double *params)
    {
        double ret = ((*params) * (1.0 - pow(base_output, 2.0)));
        if(std::isinf(ret)){
            if(ret > 0)
                return std::numeric_limits<double>::max();
            else
                return std::numeric_limits<double>::max();
        }
        return ret;
    }
};


Neuron::Neuron(uint32_t weights_count, uint32_t momentum_count, double beta, func_ptr activation, func_ptr activation_derivative)
{
    this->beta = beta;
    this->base = activation;
    this->derivative = activation_derivative;
    this->weights_count = weights_count;
    this->momentum_count = momentum_count;

    this->weights = std::vector<double>(weights_count * (1 + momentum_count), 0);

    std::mt19937_64 rng;
    // initialize the random number generator with time-dependent seed
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> unif(-0.5, 0.5);
    
    this->bias = unif(rng);
    
    for (uint32_t i{0};i<this->weights_count;i++)
        this->weights[i] = unif(rng);
}


Neuron::~Neuron() {}