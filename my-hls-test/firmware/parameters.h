#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/s6.h"
#include "weights/b6.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/s10.h"
#include "weights/b10.h"
#include "weights/w12.h"
#include "weights/b12.h"
#include "weights/s14.h"
#include "weights/b14.h"
#include "weights/w16.h"
#include "weights/b16.h"

//hls-fpga-machine-learning insert layer-config
// average_pooling2d
struct config2 : nnet::pooling2d_config {
    static const unsigned in_height = N_INPUT_1_1;
    static const unsigned in_width = N_INPUT_2_1;
    static const unsigned n_filt = N_FILT_2;
    static const unsigned stride_height = 4;
    static const unsigned stride_width = 4;
    static const unsigned pool_height = 4;
    static const unsigned pool_width = 4;

    static const unsigned filt_height = 4;
    static const unsigned filt_width = 4;
    static const unsigned n_chan = N_FILT_2;

    static const unsigned out_height = OUT_HEIGHT_2;
    static const unsigned out_width = OUT_WIDTH_2;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Average;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse = 16;
    typedef ap_fixed<16,6> accum_t;
};

// q_dense
struct config4 : nnet::dense_config {
    static const unsigned n_in = N_SIZE_1_3;
    static const unsigned n_out = N_LAYER_4;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 15;
    static const unsigned n_zeros = 1692;
    static const unsigned n_nonzeros = 55908;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<18,6> accum_t;
    typedef bias4_t bias_t;
    typedef weight4_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// batch_normalization
struct config6 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 16;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_bias_t bias_t;
    typedef batch_normalization_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// q_activation
struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 16;
    typedef ap_fixed<18,8> table_t;
};

// q_dense_1
struct config8 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned n_out = N_LAYER_8;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 16;
    static const unsigned n_zeros = 67;
    static const unsigned n_nonzeros = 1981;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<18,6> accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// batch_normalization_1
struct config10 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 16;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_1_bias_t bias_t;
    typedef batch_normalization_1_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// q_activation_1
struct relu_config11 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 16;
    typedef ap_fixed<18,8> table_t;
};

// q_dense_2
struct config12 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned n_out = N_LAYER_12;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 16;
    static const unsigned n_zeros = 12;
    static const unsigned n_nonzeros = 500;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<18,6> accum_t;
    typedef bias12_t bias_t;
    typedef weight12_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// batch_normalization_2
struct config14 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_12;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 16;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_2_bias_t bias_t;
    typedef batch_normalization_2_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// q_activation_2
struct relu_config15 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_12;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 16;
    typedef ap_fixed<18,8> table_t;
};

// q_dense_3
struct config16 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_12;
    static const unsigned n_out = N_LAYER_16;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 16;
    static const unsigned n_zeros = 33;
    static const unsigned n_nonzeros = 63;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<18,6> accum_t;
    typedef bias16_t bias_t;
    typedef weight16_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};


#endif
