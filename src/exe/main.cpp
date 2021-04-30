#include <lib/net.h>

int main()
{
    std::random_device r;
    std::default_random_engine gen(r());
    std::uniform_real_distribution<double> dis(-10.0, 10.0);

    auto myNet = Net();
    myNet.add_layer(100);
    myNet.add_layer(1);
    std::vector<double> input;
    for (uint32_t i{0}; i < 100; i++)
    {
        input.push_back(dis(gen));
    }
    std::vector<double> output;

    myNet.feed(input, output);

    for (auto &value : output)
        std::cout << value << std::endl;

    return 0;
}