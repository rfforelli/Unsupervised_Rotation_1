//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_1,
    hls::stream<layer16_t> &layer16_out,
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_1,layer16_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    const_size_out_1 = N_LAYER_16;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight4_t, 57600>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 64>(b4, "b4.txt");
        nnet::load_weights_from_txt<batch_normalization_scale_t, 64>(s6, "s6.txt");
        nnet::load_weights_from_txt<batch_normalization_bias_t, 64>(b6, "b6.txt");
        nnet::load_weights_from_txt<weight8_t, 2048>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 32>(b8, "b8.txt");
        nnet::load_weights_from_txt<batch_normalization_1_scale_t, 32>(s10, "s10.txt");
        nnet::load_weights_from_txt<batch_normalization_1_bias_t, 32>(b10, "b10.txt");
        nnet::load_weights_from_txt<weight12_t, 512>(w12, "w12.txt");
        nnet::load_weights_from_txt<bias12_t, 16>(b12, "b12.txt");
        nnet::load_weights_from_txt<batch_normalization_2_scale_t, 16>(s14, "s14.txt");
        nnet::load_weights_from_txt<batch_normalization_2_bias_t, 16>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight16_t, 96>(w16, "w16.txt");
        nnet::load_weights_from_txt<bias16_t, 6>(b16, "b16.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=900
    nnet::pooling2d_cl<input_t, layer2_t, config2>(input_1, layer2_out); // average_pooling2d
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer2_t>(layer2_out, "average_pooling2d", OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2);
#endif

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=1
    nnet::dense<layer2_t, layer4_t, config4>(layer2_out, layer4_out, w4, b4); // q_dense
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer4_t>(layer4_out, "q_dense", N_LAYER_4);
#endif

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=1
    nnet::normalize<layer4_t, layer6_t, config6>(layer4_out, layer6_out, s6, b6); // batch_normalization
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer6_t>(layer6_out, "batch_normalization", N_LAYER_4);
#endif

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=1
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out); // q_activation
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer7_t>(layer7_out, "q_activation", N_LAYER_4);
#endif

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=1
    nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // q_dense_1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer8_t>(layer8_out, "q_dense_1", N_LAYER_8);
#endif

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1
    nnet::normalize<layer8_t, layer10_t, config10>(layer8_out, layer10_out, s10, b10); // batch_normalization_1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer10_t>(layer10_out, "batch_normalization_1", N_LAYER_8);
#endif

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=1
    nnet::relu<layer10_t, layer11_t, relu_config11>(layer10_out, layer11_out); // q_activation_1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer11_t>(layer11_out, "q_activation_1", N_LAYER_8);
#endif

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1
    nnet::dense<layer11_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12); // q_dense_2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer12_t>(layer12_out, "q_dense_2", N_LAYER_12);
#endif

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=1
    nnet::normalize<layer12_t, layer14_t, config14>(layer12_out, layer14_out, s14, b14); // batch_normalization_2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer14_t>(layer14_out, "batch_normalization_2", N_LAYER_12);
#endif

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=1
    nnet::relu<layer14_t, layer15_t, relu_config15>(layer14_out, layer15_out); // q_activation_2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer15_t>(layer15_out, "q_activation_2", N_LAYER_12);
#endif

    nnet::dense<layer15_t, layer16_t, config16>(layer15_out, layer16_out, w16, b16); // q_dense_3
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer16_t>(layer16_out, "q_dense_3", N_LAYER_16);
#endif

}
