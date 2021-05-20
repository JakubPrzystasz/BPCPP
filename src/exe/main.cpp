#include <lib/net.h>

int main()
{
    pattern_set input;
    Net::read_file("INPUT_DATA.txt", input);
    //Net::open_file("OUTPUT.json");
    auto net = Net(input, {1.0,0.0});

    LearnParams params = LearnParams();
    params.init_function = InitFunction::nw;

    LearnOutput output;

    net.setup({13,6}, params);

    // for(auto &layer: net.layers){
    //     for(auto &neuron:layer.neurons){
    //         neuron.bias = 0.5;
    //         for(auto &weight: neuron.weights){
    //             weight = 0.5;
    //         }
    //     }
    // }

    output = net.train(500,0.25);
    //std::cout << static_cast<uint32_t>(output.result) << std::endl;
    //Net::save_output("OUTPUT.json", output);

    // uint32_t S1_MIN{1};
    // uint32_t S1_MAX{5};
    
    // uint32_t S2_MIN{1};
    // uint32_t S2_MAX{5};

    // uint32_t left{(S1_MAX - S1_MIN + 1) * (S2_MAX - S2_MIN + 1)};
    // uint32_t i{1};

    // for(uint32_t S1{S1_MIN};S1<=S1_MAX;S1++){
    //     for(uint32_t S2{S2_MIN};S2<=S2_MAX;S2++){
    //         net.setup({13, 6}, params);
    //         output = net.train(1000000,0.25);
    //         Net::save_output("OUTPUT.json", output);
    //         std::cout << "Iterations left: " << left - i << "   S1:" << S1 << "   S2:" << S2 << std::endl;
    //         i++;
    //     }
    // }

    //Net::close_file("OUTPUT.json");
    return 0;
}