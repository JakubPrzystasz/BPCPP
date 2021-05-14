#include <lib/net.h>
#include <sciplot/sciplot.hpp>

int main()
{
    pattern_set input;
    Net::read_file(std::string("INPUT_DATA.txt"), input);

    auto myNet = Net(input, {1.0, 0});

    LearnParams netParams = LearnParams();
    netParams.momentum_constans = 0;
    netParams.learning_accelerating_constans = 0;
    netParams.momentum_delta_vsize = 0;
    netParams.batch_size = 1;
    netParams.error_ratio = 1.04;
    netParams.learning_rate = 0.01;

    myNet.setup({6}, netParams);
    auto test = myNet.train(1000);

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