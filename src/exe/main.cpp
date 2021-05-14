#include <lib/net.h>
#include <sciplot/sciplot.hpp>
#include <chrono>

int main()
{
    pattern_set input;
    Net::read_file(std::string("INPUT_DATA.txt"), input);

    auto myNet = Net(input, {0.8, 0.2});

    LearnParams netParams = LearnParams();
    netParams.momentum_constans = 0;
    netParams.learning_accelerating_constans = 1.01;
    netParams.momentum_delta_vsize = 0;
    netParams.batch_size = 1;
    netParams.error_ratio = 1.001;
    netParams.learning_rate = 0.01;

    myNet.setup({6}, netParams);
    auto start = std::chrono::high_resolution_clock::now();
    auto test = myNet.train(1000);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  
    // To get the value of duration use the count()
    // member function on the duration object
    std::cout << duration.count() << std::endl;

    std::cout << test.train_set_SSE.back() << std::endl;

    // data_row x, target_plot, output_plot;
    // output_plot = data_row(input.size());
    // target_plot = data_row(input.size());

    // myNet.train(500, 0.1);

    // auto &out = myNet.layers.back().neurons;

    // for (uint32_t i{0}; i < input.size(); i++)
    // {
    //     myNet.feed(i);
    //     output_plot[i] = out[0].output;
    // }

    // sciplot::Plot plot0;
    // for (uint32_t i{0}; i < input.size(); i++)
    //     x.push_back(i);

    // plot0.drawDots(x, output_plot);

    // plot0.legend().hide();
    // plot0.show();

    return 0;
}