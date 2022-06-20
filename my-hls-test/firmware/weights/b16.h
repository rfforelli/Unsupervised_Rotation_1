//Numpy array shape [6]
//Min -1.125000000000
//Max 0.156250000000
//Number of zeros 2

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[6];
#else
bias16_t b16[6] = {-1.12500, 0.06250, 0.15625, 0.00000, 0.00000, -0.28125};
#endif

#endif
