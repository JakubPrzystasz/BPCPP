#include "neuron.h"

namespace ActivationFunction{
    double unipolar(double input, double *params)
    {
        return 1.0/(1.0 + pow(M_E, -((*params)*input)));
    }


    double unipolar_derivative(double base_output, double *params)
    {
        return (*params) * base_output * (1.0 - base_output);
    }


    double bipolar(double input, double *params)
    {
        return 2.0/(1.0 + pow(M_E, -1*((*params)*input))-1.0);
    }


    double bipolar_derivative(double base_output, double *params)
    {
        return (*params) * (1.0 - pow(base_output, 2.0));
    }
};