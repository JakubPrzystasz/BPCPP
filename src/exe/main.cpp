#include <lib/net.h>

int main()
{
    pattern_set input;
    Net::read_file(std::string("INPUT_DATA.txt"), input);

    auto myNet = Net(input, {0.8, 0.2});

    LearnParams netParams = LearnParams();

    myNet.setup({13, 3}, netParams);

    auto test = myNet.train(2000);

    Net::save_output("OUTPUT.json", test, Net::SaveMode::Overwrite);

    return 0;
}