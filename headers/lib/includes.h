#pragma once

#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>
#include "../matplotlib/matplotlib.h"
namespace plt = matplotlibcpp;


typedef std::vector<double> data_row;
typedef std::vector<data_row> data_set;
typedef double (*func_ptr)(double, double *);

