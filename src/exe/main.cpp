#include <lib/net.h>
#include <sciplot/sciplot.hpp>

int main()
{
    pattern_set input;
    Net::read_file(std::string("INPUT_DATA.txt"), input);

    auto myNet = Net(input, {0.8, 0.1, 0.1});

    LearnParams netParams = LearnParams();

    myNet.setup({6, 3}, netParams);







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