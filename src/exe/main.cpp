#include <lib/net.h>
#include <sciplot/sciplot.hpp>

int main()
{
    pattern_set input;
    Net::read_file(std::string("INPUT_DATA.txt"), input);

    //Define hidden layers
    std::vector<uint32_t> layers{6};

    auto myNet = Net(input);
    myNet.setup(layers, 1);

    data_row x, target_plot, output_plot;
    output_plot = data_row(input.size());
    target_plot = data_row(input.size());

    myNet.train(10000);

    auto &out = myNet.layers.back().neurons;

    for (uint32_t i{0}; i < input.size(); i++)
    {
        myNet.feed(i);
        output_plot[i] = out[0].output;
    }

    sciplot::Plot plot0;
    for (uint32_t i{0}; i < input.size(); i++)
        x.push_back(i);

    plot0.drawDots(x, output_plot);

    plot0.legend().hide();
    plot0.show();

    return 0;
}