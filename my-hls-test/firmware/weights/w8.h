//Numpy array shape [64, 32]
//Min -2.250000000000
//Max 3.343750000000
//Number of zeros 67

#ifndef W8_H_
#define W8_H_

#ifndef __SYNTHESIS__
weight8_t w8[2048];
#else
weight8_t w8[2048] = {0.06250, -0.78125, -0.03125, -0.12500, -0.12500, 0.06250, -0.78125, -0.87500, 0.53125, 0.37500, -0.31250, 0.31250, 0.00000, 0.12500, -1.06250, 0.18750, -0.37500, -0.31250, 0.03125, 0.15625, 0.12500, -0.31250, -0.28125, -0.21875, 0.37500, -0.65625, -0.09375, 0.28125, -0.75000, -0.43750, -0.21875, 1.40625, 0.31250, 0.18750, -0.93750, 0.18750, 0.18750, 0.28125, 0.03125, 0.87500, -0.12500, 0.56250, -0.03125, -0.12500, 0.00000, -0.21875, 0.09375, 0.40625, -0.15625, 0.03125, -0.25000, 0.06250, -0.81250, -0.78125, 0.25000, -0.03125, 0.09375, 0.31250, 0.34375, 0.06250, 0.50000, 1.40625, -0.03125, -0.03125, -1.40625, -0.81250, -0.21875, -0.09375, -0.71875, -0.68750, 1.00000, -0.18750, -0.50000, 0.12500, 0.03125, 0.03125, -0.25000, 0.06250, 1.12500, -0.81250, 0.40625, -0.81250, -0.25000, 0.06250, -0.21875, 0.15625, 0.00000, -0.06250, 0.09375, 0.43750, -1.15625, -1.50000, 0.25000, 0.75000, 0.68750, -0.75000, 0.12500, 0.09375, -0.09375, -0.09375, -1.75000, -0.93750, 0.15625, -0.06250, -0.21875, -0.93750, -0.03125, -0.28125, 0.18750, 0.06250, 0.09375, -0.12500, 0.21875, -0.18750, -0.40625, -0.34375, -0.75000, -0.15625, -0.28125, -0.09375, -0.12500, -0.65625, 0.12500, -0.28125, 0.68750, -0.18750, -0.18750, 0.00000, 0.28125, 0.87500, -0.12500, -0.25000, -0.31250, 0.09375, -0.81250, -0.62500, -0.37500, 0.09375, 0.18750, 0.75000, 0.09375, 0.56250, -0.87500, -0.06250, 0.28125, -0.12500, 0.43750, 0.03125, 0.18750, -1.06250, 1.15625, -0.43750, -0.03125, -1.37500, 0.06250, -0.06250, -1.46875, -0.40625, 0.46875, 1.87500, -0.03125, 0.31250, 0.93750, -0.06250, -0.21875, 0.15625, -0.06250, 0.43750, -0.15625, 0.06250, 0.03125, 0.09375, 0.75000, 0.37500, 0.03125, 0.15625, 0.37500, 0.25000, 0.18750, 0.21875, 0.00000, -0.84375, -0.06250, 0.00000, 0.50000, 0.28125, 0.34375, 0.03125, 0.34375, -0.03125, 0.87500, -0.06250, 0.46875, 0.34375, -0.34375, -1.12500, 0.21875, -0.65625, 0.40625, 0.46875, -0.53125, -0.15625, 1.03125, 0.25000, 0.18750, 0.09375, 0.62500, -0.25000, 0.50000, 0.00000, 0.75000, -0.25000, -0.09375, -0.06250, 0.34375, -0.43750, 0.06250, 1.15625, 0.21875, 0.06250, 0.06250, 0.53125, 0.34375, -1.12500, -0.09375, -0.46875, -0.40625, -0.15625, -0.28125, 0.34375, -0.15625, -0.59375, 0.03125, 0.18750, -0.43750, 0.18750, 0.93750, -0.34375, 0.18750, -0.84375, -0.59375, 0.50000, 0.09375, -0.90625, 0.53125, -0.09375, 0.09375, -0.18750, -0.90625, 0.03125, -0.12500, -0.15625, 0.37500, -0.65625, 0.31250, -0.06250, -0.31250, 0.65625, 0.56250, 0.75000, -0.03125, -0.15625, 0.00000, 0.59375, 0.50000, -0.68750, -0.31250, -0.50000, -0.09375, 0.12500, 0.18750, 0.25000, -0.15625, 0.06250, 0.53125, 0.15625, -0.06250, -0.34375, 0.56250, -0.46875, 0.18750, 0.46875, 0.40625, -1.90625, -0.56250, 0.03125, 0.09375, -1.28125, 0.09375, -0.53125, 0.50000, 0.28125, 0.15625, 0.40625, -0.21875, -0.65625, 0.65625, 0.21875, -0.03125, 0.21875, -0.09375, 0.09375, -0.12500, -1.03125, -1.06250, -0.15625, -0.06250, -0.18750, 0.65625, 0.25000, 0.15625, -0.06250, -1.25000, -0.34375, -1.09375, 0.00000, -0.21875, -0.87500, 0.56250, -0.40625, -1.03125, -0.53125, 0.15625, 0.28125, 0.06250, 0.75000, -0.15625, -0.31250, 0.00000, -0.06250, 0.15625, 0.06250, 0.18750, 0.62500, 0.00000, -0.43750, 0.12500, 0.00000, 0.59375, -0.09375, -0.06250, -0.18750, 0.43750, 0.81250, -0.25000, 0.81250, -0.28125, -1.81250, 0.28125, -0.28125, 0.84375, -1.46875, -0.28125, 0.15625, 0.34375, -1.06250, 0.12500, -0.31250, -0.28125, -0.50000, 0.18750, -0.31250, -0.34375, -0.43750, -0.28125, -0.18750, 0.12500, 0.50000, 0.00000, 0.78125, 0.50000, 0.15625, 0.12500, -0.21875, -0.25000, -0.21875, 0.06250, -0.34375, 0.40625, -0.12500, 0.34375, -1.21875, -0.12500, 0.15625, 0.03125, -0.37500, 0.34375, 0.18750, 0.28125, 0.43750, 0.18750, -0.34375, -0.87500, -0.40625, -1.28125, -0.90625, -0.34375, -0.12500, 0.37500, -0.09375, -0.25000, -0.28125, 0.31250, 0.59375, 0.43750, -0.03125, -0.12500, 0.09375, 0.09375, 0.59375, -0.28125, -0.56250, 0.25000, -0.18750, 0.15625, -0.84375, -0.25000, -0.43750, -0.31250, -0.18750, 0.31250, -0.12500, -0.71875, -0.50000, -0.62500, -0.56250, 0.50000, 0.21875, -0.21875, -0.84375, -0.37500, -0.56250, 0.21875, -0.06250, -0.75000, 0.62500, -0.93750, 0.15625, 0.28125, 0.56250, -0.71875, -0.93750, -0.90625, 0.40625, -0.18750, -0.15625, -1.12500, 0.53125, 0.43750, 0.03125, -0.06250, -0.09375, 0.03125, 0.50000, -0.34375, -1.53125, 0.25000, 0.71875, 0.93750, 0.18750, 0.12500, 0.62500, -1.21875, -0.12500, -0.21875, 0.56250, 0.03125, 0.09375, -0.06250, -0.37500, 0.59375, 0.28125, 0.06250, -0.34375, 0.18750, 1.25000, 0.15625, -0.25000, -0.43750, 0.12500, 0.37500, -0.03125, 0.81250, -0.09375, 0.03125, 0.15625, -0.12500, -0.28125, -0.03125, 0.21875, 0.25000, 0.09375, 0.90625, -0.34375, 0.15625, -0.90625, -0.56250, 0.15625, 1.31250, -0.28125, 0.09375, 0.06250, -0.09375, 0.03125, 0.25000, -0.43750, 0.15625, 0.06250, 0.53125, 0.81250, 0.56250, 0.06250, 0.25000, 0.31250, -0.31250, -0.21875, 0.40625, 1.15625, -0.40625, -0.50000, -0.06250, 0.25000, -0.40625, 0.15625, 0.21875, -0.25000, -0.03125, -0.15625, 0.15625, -0.12500, 0.25000, -0.18750, 0.18750, -0.12500, 0.78125, 0.56250, -0.34375, -0.37500, 0.53125, 0.43750, 0.81250, -0.46875, -0.50000, 1.53125, 0.18750, 0.15625, 0.56250, -0.06250, 0.25000, -0.28125, 0.06250, 0.06250, -0.15625, 0.03125, 0.37500, -0.37500, -0.06250, 0.18750, 0.21875, 0.46875, 0.00000, -0.18750, 0.06250, 0.21875, 0.56250, -0.43750, -0.12500, -0.09375, 0.09375, -0.25000, 0.18750, -0.06250, -0.06250, 0.59375, -0.28125, -0.40625, -0.75000, -0.81250, -0.03125, 0.15625, 0.12500, -0.71875, 0.34375, 0.03125, -0.18750, 0.00000, -0.81250, -0.12500, 0.00000, -0.09375, 0.68750, -0.84375, 0.28125, -0.37500, 0.90625, -0.25000, -0.06250, -0.15625, 0.56250, -0.06250, -0.09375, 1.09375, -0.90625, -1.40625, 0.28125, 0.40625, 0.46875, -0.62500, 0.00000, 0.03125, 0.37500, -0.96875, -0.46875, -0.78125, -0.31250, 0.00000, -0.03125, -0.46875, -0.31250, 0.15625, -0.15625, -0.46875, 0.37500, 0.12500, -0.12500, -0.15625, -0.40625, 0.12500, 0.96875, 0.12500, -0.06250, -0.09375, -0.21875, -0.06250, -0.18750, -0.09375, 0.03125, -0.15625, 0.37500, -0.40625, -0.40625, 0.21875, 0.06250, -0.28125, 0.37500, 0.50000, -1.25000, -0.37500, -0.81250, 0.03125, 0.12500, 1.06250, 0.03125, -0.59375, -1.28125, 0.21875, 0.15625, 0.93750, 0.06250, -0.09375, -0.06250, -0.06250, 0.03125, -0.21875, 0.28125, -0.84375, 0.18750, 0.31250, 0.06250, -0.90625, -0.21875, -0.53125, -0.50000, 0.31250, 0.93750, 0.03125, -0.18750, 0.31250, -0.06250, 0.46875, 0.25000, 0.31250, -0.09375, 0.28125, -1.18750, 1.03125, -0.31250, 0.00000, -0.09375, 0.18750, 1.15625, 0.59375, 0.28125, 0.00000, 0.15625, 0.09375, 0.40625, 1.06250, 0.03125, -0.03125, 0.50000, 0.56250, -1.46875, -0.28125, -0.37500, -1.18750, 0.09375, 0.09375, -1.06250, -0.31250, 1.09375, 2.06250, -0.59375, 0.09375, -0.50000, -0.59375, -0.12500, -0.21875, 1.34375, -0.09375, 0.12500, -0.34375, 0.12500, 0.21875, 0.03125, -0.06250, -0.03125, 0.03125, 0.06250, 0.65625, -0.34375, -0.65625, -0.12500, 0.71875, 0.21875, -0.31250, 0.03125, -0.03125, 0.06250, 0.21875, -0.37500, -0.06250, 0.06250, 0.03125, -0.21875, -0.75000, 0.18750, 0.09375, -0.25000, -0.34375, 0.03125, -0.59375, -0.03125, -0.21875, -0.28125, 0.21875, -0.25000, 0.09375, -0.03125, -0.06250, -0.21875, -0.28125, -0.40625, -0.03125, -0.09375, -0.09375, 0.18750, -0.03125, -0.68750, 0.12500, 0.09375, 0.62500, -0.81250, -0.59375, 0.18750, -0.09375, -0.53125, -0.34375, -0.21875, -0.31250, -0.75000, 0.09375, 0.18750, -0.84375, 0.34375, -0.71875, 0.18750, 0.15625, 0.40625, -0.25000, 0.34375, -0.46875, 0.15625, 0.09375, -0.96875, 0.46875, 0.18750, 0.12500, 0.15625, 0.03125, 0.46875, -0.31250, 0.18750, -1.21875, -1.46875, -1.03125, -0.65625, -0.09375, -0.31250, -0.96875, 0.03125, 0.31250, -0.28125, -0.78125, 0.31250, -0.34375, -0.09375, -0.43750, -0.21875, 0.31250, 0.28125, 0.15625, 0.31250, 0.28125, -0.25000, 0.06250, -0.96875, 0.15625, -0.12500, -0.15625, 0.43750, 0.12500, -0.12500, 1.50000, -0.09375, -0.34375, -0.34375, 0.31250, -0.31250, -0.40625, 0.18750, 0.62500, 0.50000, 0.21875, -0.09375, 0.34375, -0.21875, 0.06250, -0.12500, 0.59375, -0.09375, 0.12500, -0.12500, -0.18750, 0.12500, -0.34375, 0.28125, -0.21875, 0.40625, -0.09375, -0.31250, -0.15625, -0.09375, 0.09375, -0.62500, 0.12500, 0.71875, 0.00000, 0.18750, 0.15625, 0.06250, -0.81250, 0.21875, -0.53125, -0.31250, 0.00000, 1.31250, 0.18750, 0.00000, 0.21875, -0.15625, -0.03125, 0.34375, -0.43750, 0.28125, -0.40625, 0.06250, -0.06250, 0.25000, 0.75000, 0.15625, 0.03125, 0.37500, -0.09375, 0.00000, 0.15625, -0.56250, 0.28125, 0.00000, 0.12500, -0.21875, 0.56250, 0.28125, -1.25000, -0.43750, 0.15625, 0.46875, 0.18750, 0.12500, 0.09375, 0.21875, 0.06250, 0.12500, -0.31250, 0.37500, -0.18750, -0.06250, 0.37500, -0.12500, -0.25000, -1.09375, 0.40625, 0.12500, -0.62500, 0.43750, 0.21875, 0.15625, -0.09375, 1.03125, -0.15625, -1.46875, 0.25000, 0.25000, -0.34375, -0.18750, -0.56250, -0.31250, 0.37500, 0.28125, 0.00000, -2.06250, -1.31250, -0.09375, 0.18750, 0.37500, 0.06250, 0.62500, 0.56250, -0.31250, -0.31250, -0.37500, 0.03125, -0.46875, 0.78125, 0.03125, 0.34375, -0.15625, 0.00000, 0.18750, 0.15625, 0.03125, 0.28125, -0.56250, -0.87500, 0.37500, -0.09375, 0.40625, 0.28125, 0.21875, 0.31250, 0.40625, -0.09375, 0.06250, 0.43750, 0.75000, 0.03125, 0.25000, -1.09375, 1.06250, -0.25000, -0.37500, -0.25000, 0.56250, 0.18750, -0.43750, 1.37500, 0.68750, 0.25000, 0.50000, 0.40625, 0.59375, 0.81250, -0.28125, 0.03125, -0.53125, 0.37500, 0.06250, 0.84375, -0.03125, -0.65625, -0.03125, -0.21875, -0.40625, -0.28125, 0.00000, -0.15625, 0.15625, -0.06250, -0.21875, 0.25000, -1.03125, -0.40625, -0.31250, -0.15625, -0.25000, -0.21875, -0.34375, 0.06250, 0.21875, -0.25000, -0.09375, -0.62500, 0.56250, 0.09375, -0.03125, -0.40625, -0.50000, -0.50000, 1.12500, 0.50000, 0.59375, 0.15625, -0.12500, 0.00000, -0.43750, -0.68750, 0.21875, -0.18750, 0.65625, -0.03125, 0.34375, -0.37500, 0.87500, -0.28125, -0.21875, -0.28125, 0.78125, 0.43750, -0.65625, 0.18750, 0.50000, 0.06250, 0.75000, 0.28125, 0.90625, 0.56250, 0.06250, 0.43750, -0.46875, -0.06250, 0.00000, 1.15625, -0.18750, -0.46875, 0.40625, 0.40625, -0.28125, -0.31250, -0.50000, 0.00000, 0.31250, -0.09375, -0.03125, 0.28125, -0.46875, 0.00000, 0.75000, 0.12500, -0.43750, -0.15625, 0.06250, 0.43750, 0.28125, -0.25000, 0.21875, 0.53125, 0.59375, 0.03125, 0.90625, -0.03125, 0.31250, -0.09375, -0.37500, 0.34375, 0.09375, 0.03125, 0.34375, -0.25000, -0.40625, -0.06250, 0.00000, 0.09375, 0.40625, -0.06250, -0.84375, 0.28125, -0.06250, 0.09375, -0.18750, -0.25000, 0.34375, 0.50000, -0.06250, 0.68750, 0.00000, 0.12500, 0.78125, 0.00000, -0.09375, -1.25000, 0.09375, -0.37500, -0.96875, -0.25000, 0.25000, -0.28125, -0.21875, -1.96875, 0.12500, -0.15625, 0.43750, 0.25000, -0.21875, 0.46875, 0.09375, -0.21875, -0.15625, -0.84375, 0.18750, 0.53125, 0.59375, 0.71875, 0.09375, 0.25000, -0.71875, 0.59375, 0.03125, 0.09375, -1.37500, -0.59375, 0.09375, -0.59375, 0.34375, 0.81250, 0.06250, -0.18750, -0.06250, -0.25000, -1.59375, 1.00000, -1.12500, -0.12500, 0.53125, 0.46875, 0.40625, 0.31250, -2.03125, 0.34375, -0.62500, 0.18750, -0.03125, 0.06250, 0.00000, 0.00000, -0.56250, -0.25000, -0.28125, -0.53125, 0.50000, 1.09375, -0.25000, -1.40625, -0.87500, 0.59375, -0.37500, -0.12500, -1.00000, 0.34375, 0.68750, 0.56250, 0.03125, 0.06250, 0.50000, 0.00000, 0.09375, 0.09375, 0.59375, 1.31250, -0.78125, -0.09375, -0.50000, 0.06250, -0.15625, 0.31250, 0.90625, 0.15625, -0.06250, 0.00000, -0.43750, 0.46875, 0.34375, 0.03125, -0.81250, 0.06250, -0.21875, -0.18750, -0.34375, -0.43750, 0.00000, 0.18750, 0.12500, -0.06250, -0.34375, -0.09375, -0.87500, 0.34375, 0.28125, 0.43750, -0.21875, 1.09375, 0.81250, -0.37500, -0.21875, -0.31250, -0.06250, 0.09375, 0.18750, -1.00000, -0.15625, 0.46875, -0.65625, -0.09375, -0.46875, -0.28125, 0.40625, 0.46875, 0.06250, 0.56250, 0.43750, -0.37500, -1.03125, -0.18750, -0.06250, -0.21875, 0.09375, 0.25000, -0.46875, 0.00000, 0.21875, 0.06250, -1.15625, 0.15625, -0.25000, -0.25000, 0.09375, -0.31250, -1.12500, -0.03125, 0.62500, 0.21875, -0.28125, -0.03125, -0.28125, 0.43750, -0.03125, -0.12500, -1.18750, -0.50000, -1.31250, -0.12500, -0.46875, -0.09375, 0.34375, 0.46875, 0.21875, -0.28125, 0.03125, 0.81250, -0.75000, 0.25000, -0.53125, 0.59375, -0.65625, -0.46875, 0.21875, -0.09375, -0.03125, 1.25000, 0.37500, 0.25000, -0.21875, 0.37500, -0.06250, -0.71875, -0.03125, 0.06250, -0.21875, -0.90625, 0.28125, 0.03125, -0.03125, -0.09375, -0.78125, 0.00000, -0.53125, -0.09375, 0.34375, -0.43750, -0.06250, -1.03125, -1.78125, 0.15625, 0.28125, 0.31250, 0.46875, -0.21875, -0.56250, 0.25000, 0.06250, 0.00000, -0.03125, 0.59375, 0.96875, 0.53125, 0.03125, 0.00000, 0.31250, 0.15625, 0.00000, 0.12500, -0.06250, 0.15625, -0.37500, 0.25000, 0.46875, -0.78125, 0.15625, 0.31250, 0.62500, -0.09375, -0.43750, -0.06250, -0.43750, 0.34375, -0.03125, -0.12500, 0.34375, -0.18750, -0.43750, 0.03125, 0.18750, -0.12500, 0.71875, -0.12500, 0.06250, 0.21875, -0.03125, 0.53125, -0.21875, 0.15625, -0.46875, 0.40625, 0.03125, -0.46875, -0.50000, -0.28125, 0.81250, -0.75000, -0.59375, -0.46875, -0.18750, -1.34375, -0.18750, 0.21875, -0.43750, 0.78125, 0.25000, 0.18750, -1.75000, -1.53125, -0.28125, 0.40625, 0.46875, 0.28125, 0.71875, 0.12500, -0.18750, 0.09375, -0.06250, 0.28125, 0.00000, -0.03125, -0.03125, 0.28125, -0.50000, -0.28125, -0.15625, 0.37500, 0.31250, 1.53125, -0.31250, -0.50000, -0.28125, 0.03125, -0.18750, 0.46875, -0.12500, -0.28125, 0.62500, -0.53125, -0.15625, 0.00000, 0.59375, 0.03125, 0.34375, 0.00000, -0.59375, 0.25000, 0.34375, 0.00000, -0.50000, -0.21875, 0.34375, 0.56250, -0.15625, 0.87500, 0.18750, 0.03125, -0.25000, -1.81250, -0.25000, -0.75000, 0.06250, -0.25000, -0.31250, -0.78125, 0.18750, -1.25000, 0.15625, 0.09375, 0.37500, 0.25000, 0.56250, 0.50000, -0.12500, -0.78125, 0.00000, -1.03125, 0.59375, -0.25000, -0.12500, -0.21875, 0.09375, 0.03125, -1.15625, 0.12500, -0.06250, -0.12500, 0.21875, -0.59375, -0.56250, -0.06250, 0.15625, 1.25000, -0.21875, -0.34375, 0.31250, 0.81250, -0.90625, -1.90625, -0.03125, 0.46875, 0.68750, -0.93750, 0.06250, 0.25000, -1.34375, 0.25000, 0.15625, -0.40625, 0.06250, -0.21875, 0.34375, -0.53125, -0.65625, 0.34375, 0.06250, -0.90625, 0.06250, 1.15625, 0.28125, -0.81250, -0.25000, 1.37500, 0.28125, -0.09375, 0.00000, 0.00000, -0.21875, 0.12500, -0.03125, 0.37500, 0.21875, 0.28125, -0.12500, 0.06250, 1.28125, 0.81250, 0.28125, 0.09375, -0.18750, 0.03125, 0.00000, -0.50000, 0.18750, -0.43750, -0.03125, 0.03125, 0.00000, 0.25000, -0.15625, -0.03125, 0.56250, 0.18750, -0.40625, -0.09375, 0.25000, 3.34375, -0.50000, -0.31250, 0.28125, 0.68750, 0.37500, -0.15625, 0.06250, 0.25000, 0.37500, 0.21875, 0.09375, 0.00000, 0.50000, 0.09375, 0.09375, -0.25000, 0.40625, -0.37500, -0.18750, 0.12500, 0.37500, -0.03125, -0.34375, 0.43750, 0.31250, 0.21875, 0.12500, 0.28125, 0.46875, -2.15625, -0.06250, 0.28125, 0.34375, -0.06250, 0.06250, -0.09375, 0.12500, -0.96875, -0.28125, -0.06250, -0.18750, -0.15625, -0.09375, 0.15625, 0.25000, 0.81250, 0.03125, 0.06250, 0.15625, -0.25000, 0.31250, -0.12500, -0.25000, -0.28125, 0.15625, -0.43750, 0.18750, -0.15625, 0.21875, -0.37500, 0.28125, -0.25000, 0.15625, -1.34375, 0.03125, -0.25000, -0.12500, -0.46875, 0.28125, 0.28125, -0.09375, 0.59375, 0.68750, 0.03125, 0.18750, -0.31250, 0.43750, 0.18750, -0.03125, -0.15625, -0.46875, -0.12500, -0.12500, 0.81250, -0.59375, 0.62500, -0.59375, -0.21875, 0.25000, 0.25000, 0.25000, 0.43750, -0.21875, 0.62500, 1.09375, -0.50000, -0.71875, 0.06250, -0.25000, 0.40625, 0.12500, 0.15625, 0.12500, 0.25000, 0.28125, 0.25000, -2.25000, -0.06250, -0.62500, -0.31250, 0.46875, 0.03125, 0.53125, -0.03125, -0.31250, 0.12500, -0.90625, -0.09375, 0.06250, -0.06250, 0.06250, -0.15625, 0.21875, 0.81250, -1.03125, 0.12500, -0.34375, -0.25000, 0.25000, -0.31250, -0.09375, 0.34375, 0.12500, 0.81250, -1.15625, 0.18750, 0.28125, -1.78125, 0.06250, -0.43750, 0.21875, 0.03125, -0.34375, -1.00000, 0.15625, -0.03125, -0.28125, 0.53125, 0.65625, 0.81250, -0.28125, 0.09375, 0.68750, 0.59375, 0.34375, -0.06250, 0.28125, -1.50000, -0.34375, -0.09375, -0.21875, 0.50000, -0.03125, 0.09375, -0.15625, -0.59375, -0.59375, 0.37500, -0.09375, -0.40625, -0.96875, -0.46875, 0.37500, -0.03125, -0.15625, -0.21875, -0.43750, -0.21875, 0.06250, 0.71875, -0.31250, 0.15625, -0.25000, 0.06250, 0.28125, -0.15625, -0.50000, -0.31250, 0.46875, -0.31250, 0.21875, 1.21875, -0.03125, -0.18750, 0.15625, 0.43750, -1.21875, 0.09375, 0.15625, -0.09375, 0.40625, -0.15625, 0.40625, 0.15625, -1.12500, 0.31250, -0.65625, 0.18750, 0.53125, -0.12500, 0.03125, -0.84375, -0.53125, 0.37500, -0.03125, -1.93750, 0.65625, 0.50000, 0.18750, -0.71875, 0.46875, -0.09375, 0.25000, 0.09375, -1.03125, 0.37500, 0.50000, 0.50000, 0.12500, -0.21875, 0.21875, 0.31250, 0.09375, 0.06250, -0.09375, 0.62500, -0.25000, 1.12500, -0.06250, 0.50000, 0.15625, 0.28125, 0.59375, -0.31250, -0.15625, -0.06250, 0.62500, 0.00000, 0.25000, -0.06250, -0.21875, -0.87500, 0.00000, -0.50000, 0.34375, -0.81250, 0.31250, 0.21875, -0.81250, -1.00000, 0.00000, -0.15625, -0.53125, -0.15625, -0.56250, -1.28125, -0.59375, -0.46875, -0.15625, 0.56250, -0.43750, 0.21875, 0.18750, 0.00000, 0.18750, 0.09375, -0.31250, -0.15625, 0.37500, 1.21875, 0.00000, -0.75000, -0.28125, 0.06250, -0.59375, -0.18750, -0.21875, -0.31250, -0.06250, 0.15625, 0.15625, -1.71875, 0.25000, -0.68750, -1.31250, -0.59375, 0.28125, 0.31250, -0.21875, -0.12500, -0.59375, -0.06250, 0.18750, -0.50000, -0.96875, 0.00000, -0.31250, 0.31250, -0.09375, 0.15625, -0.09375, -0.34375, -0.40625, 0.18750, -0.68750, -0.34375, -0.12500, -0.06250, -0.28125, -0.93750, -0.81250, -0.71875, 0.56250, 0.18750, 0.34375, 0.59375, -0.81250, 0.31250, -0.21875, -0.34375, 0.46875, -0.06250, 0.59375, -0.68750, 0.46875, -1.28125, -0.25000, -0.15625, -0.53125, 0.40625, 1.03125, 0.21875, -0.40625, -0.03125, 0.28125, -0.65625, 0.50000, 0.34375, 0.21875, -0.43750, -0.09375, 0.25000, 0.09375, 0.12500, -1.15625, -0.71875, 0.65625, -0.62500, 0.15625, 0.06250, 0.03125, -0.21875, -0.75000, 0.31250, 0.68750, 0.06250, 0.00000, -0.65625, -0.03125, -0.71875, -0.96875, 0.28125, -0.65625, -0.40625, 0.12500, -0.87500, 0.21875, -0.65625, 0.03125, -0.12500, 0.00000, -0.56250, -0.15625, 0.53125, -0.56250, -0.25000, -0.31250, 0.09375, -0.21875, 1.00000, 0.34375, 0.28125, 0.96875, 0.90625, 0.28125, 0.46875, 0.06250, 0.21875, -0.03125, 0.09375, -0.56250, -0.50000, -0.28125, 0.09375, 0.25000, -0.15625, 0.84375, -0.12500, 0.25000, -0.09375, 0.68750, 0.15625, -0.15625, -0.46875, 0.37500, 0.37500, 0.50000, 0.15625, -0.06250, 0.53125, 0.81250, 0.28125, 0.15625, 0.25000, -0.50000, 0.28125, -0.03125, 0.84375, 0.06250, 0.25000, 0.50000, 0.65625, 0.62500, -1.25000, -0.09375, -1.03125, -0.15625, -0.25000, 0.46875, -0.12500, 0.56250, -0.43750, 0.31250, 0.00000, -0.03125, -0.62500, 0.37500, 0.15625, -0.03125, 0.12500, 0.56250, -0.18750, 0.46875, 0.37500, -1.00000, 0.15625, 0.06250, -0.68750, -0.12500, -0.15625, 0.78125, 0.18750, 0.31250, -0.18750, 0.50000, -0.21875, 0.28125, 0.28125, -0.06250, 0.90625, -0.09375, 1.75000, 0.12500, 0.21875, 0.28125, 0.53125, 0.53125, -0.09375, 0.31250, 0.34375, 0.09375, 0.00000, 0.40625, 0.50000, -0.09375, -0.81250, 0.18750, -0.03125, -0.15625, -0.37500, 0.62500, 0.31250, 0.65625, -0.12500, 0.00000, 0.18750, -0.84375, -0.12500, -0.15625, -0.21875, 0.00000, -0.09375, -0.71875, -1.21875, 0.09375, -0.28125, 0.56250, -0.15625, 0.59375, 0.03125};
#endif

#endif
