#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 120
#define N_INPUT_2_1 120
#define N_INPUT_3_1 1
#define OUT_HEIGHT_2 30
#define OUT_WIDTH_2 30
#define N_FILT_2 1
#define N_SIZE_1_3 900
#define N_LAYER_4 64
#define N_LAYER_8 32
#define N_LAYER_12 16
#define N_LAYER_16 6

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<18,6> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> input_t;
typedef ap_fixed<16,6> average_pooling2d_default_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer2_t;
typedef nnet::array<ap_fixed<18,6>, 64*1> layer4_t;
typedef ap_fixed<12,7> weight4_t;
typedef ap_fixed<12,7> bias4_t;
typedef nnet::array<ap_fixed<18,6>, 64*1> layer6_t;
typedef ap_fixed<16,6> batch_normalization_scale_t;
typedef ap_fixed<16,6> batch_normalization_bias_t;
typedef nnet::array<ap_ufixed<8,0>, 64*1> layer7_t;
typedef nnet::array<ap_fixed<18,6>, 32*1> layer8_t;
typedef ap_fixed<12,7> weight8_t;
typedef ap_fixed<12,7> bias8_t;
typedef nnet::array<ap_fixed<18,6>, 32*1> layer10_t;
typedef ap_fixed<16,6> batch_normalization_1_scale_t;
typedef ap_fixed<16,6> batch_normalization_1_bias_t;
typedef nnet::array<ap_ufixed<8,0>, 32*1> layer11_t;
typedef nnet::array<ap_fixed<18,6>, 16*1> layer12_t;
typedef ap_fixed<12,7> weight12_t;
typedef ap_fixed<12,7> bias12_t;
typedef nnet::array<ap_fixed<18,6>, 16*1> layer14_t;
typedef ap_fixed<16,6> batch_normalization_2_scale_t;
typedef ap_fixed<16,6> batch_normalization_2_bias_t;
typedef nnet::array<ap_ufixed<8,0>, 16*1> layer15_t;
typedef nnet::array<ap_fixed<18,6>, 6*1> layer16_t;
typedef ap_fixed<12,7> weight16_t;
typedef ap_fixed<12,7> bias16_t;

#endif
