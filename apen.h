#ifndef APEN_H
#define APEN_H

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

// GRID SIZE*BLOCK SIZE should equal number of samples (np=256).
#define GRID_SIZE 8
#define BLOCK_SIZE 32

double apen_correlation (int np, int32_t *x, unsigned int m, double r);
void apen(int np, int32_t *x, float *a, unsigned int m, double r);

#endif
