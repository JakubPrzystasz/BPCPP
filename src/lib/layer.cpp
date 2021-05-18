#include "layer.h"

Layer::Layer(uint32_t neurons, uint32_t inputs, LearnParams params)
{
    this->learn_parameters = params;

    this->inputs = inputs;

    this->neurons = std::vector<Neuron>(neurons, Neuron(inputs, this->learn_parameters));

    this->learn_parameters.init_function(*this);
}

namespace InitFunction
{
    double __unifrom_random(rand_range range, uint64_t seed)
    {
        std::seed_seq ss{uint32_t(seed & 0xffffffff), uint32_t(seed >> 32)};
        std::mt19937_64 rng;
        rng.seed(ss);
        std::uniform_real_distribution<double> unif(static_cast<double>(std::get<0>(range)), static_cast<double>(std::get<1>(range)));
        return unif(rng);
    }

    void rand(Layer &layer)
    {
        uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        for (auto &neuron : layer.neurons)
        {
            neuron.bias = __unifrom_random(neuron.learn_parameters.bias_range, timeSeed);
            for (auto &weight : neuron.weights)
                weight = __unifrom_random(neuron.learn_parameters.weights_range, timeSeed);
        }
    }

    void const_rand(Layer &layer)
    {
        for (auto &neuron : layer.neurons)
        {
            neuron.bias = __unifrom_random(neuron.learn_parameters.bias_range, 123456789ULL);
            for (auto &weight : neuron.weights)
                weight = __unifrom_random(neuron.learn_parameters.weights_range, 123456789ULL);
        }
    }

    void nw(Layer &layer)
    {
        uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        //Compute scaling factor
        double theta = 0.7 * std::pow(static_cast<double>(layer.inputs), 1.0 / static_cast<double>(layer.neurons.size()));
        //Initialize the wights and biases for each neuron at random, eg. U(-0.5,0.5)
        rand(layer);

        for (auto &neuron : layer.neurons)
        {
            double eta = std::sqrt(std::accumulate(VEC_RANGE(neuron.weights), 0.0, accumulate::square<double>()));
            for (auto &weight : neuron.weights)
                weight = (theta * weight) / eta;
            neuron.bias = __unifrom_random(rand_range(-1.0 * theta, theta), timeSeed);
        }
    }

}
