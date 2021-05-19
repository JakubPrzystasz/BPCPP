#include <lib/net.h>

int main()
{
    pattern_set input;
    Net::read_file("INPUT_DATA.txt", input);
    Net::open_file("OUTPUT.json");
    auto net = Net(input, {0.8, 0.2});

    LearnParams params = LearnParams();
    params.init_function = InitFunction::const_rand;
    LearnOutput output;

    uint32_t S1_MIN{1};
    uint32_t S1_MAX{30};
    
    uint32_t S2_MIN{1};
    uint32_t S2_MAX{30};

    uint32_t left{(S1_MAX - S1_MIN + 1) * (S2_MAX - S2_MIN + 1)};
    uint32_t i{1};

    for(uint32_t S1{S1_MIN};S1<=S1_MAX;S1++){
        for(uint32_t S2{S2_MIN};S2<=S2_MAX;S2++){
            net.setup({S1, S2}, params);
            output = net.train(10000,0.001);
            Net::save_output("OUTPUT.json", output);
            std::cout << "Iterations left: " << left - i << "   S1:" << S1 << "   S2:" << S2 << std::endl;
            i++;
        }
    }

    Net::close_file("OUTPUT.json");
    return 0;
}