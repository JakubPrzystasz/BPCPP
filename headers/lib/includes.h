#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
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

typedef std::vector<double> data_row;
typedef std::vector<data_row> data_set;
typedef double (*func_ptr)(double, double *);

double random_value(double min,double max);