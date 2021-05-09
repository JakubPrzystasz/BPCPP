#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <limits>
#include <cstring>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>

class Layer;
class Neuron;
class Net;
struct Pattern;

typedef std::vector<double> data_row;
typedef std::vector<data_row> data_set;
typedef std::vector<Pattern> pattern_set;
typedef std::pair<double,double> rand_range;
typedef double (*func_ptr)(double, double *);

struct Pattern{
    /**
     * Input vector
     */
    data_row input;

    /**
     *Output vector
     */
    data_row output;
};


double random_value(rand_range &range);