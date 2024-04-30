#include "stencil/solve.h"

#include <assert.h>
#include <stdio.h>
// #include <math.h>

f64 square_and_multiply(f64 x, usz n)
{
  f64 y;
  // n should be unsigned
  // if (n < 0)
  // {
  //   x = 1 / x;
  //   n = -n;
  // }

  if (n == 0)
    return 1;

  y = 1;

  while (n > 1)
  {
    if ((n & 1) != 0)
    {
      y = x * y;
      n = n - 1;
    }
    x = x * x;
    n = n >> 1; // Divided by 2
  }

  return x * y;
}

void solve_jacobi(mesh_t* A, mesh_t const* B, mesh_t* C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz const dim_x = A->dim_x;
    usz const dim_y = A->dim_y;
    usz const dim_z = A->dim_z;

    f64 o_values[STENCIL_ORDER];
    for (usz o = 1; o <= STENCIL_ORDER; ++o) {
      o_values[o - 1] = 1 / square_and_multiply(17.0, o);
    }

    f64 tmp;

    #pragma omp parallel for schedule(runtime)
    for (usz i = STENCIL_ORDER; i < dim_x - STENCIL_ORDER; i+=20) {
        for (usz j = STENCIL_ORDER; j < dim_y - STENCIL_ORDER; j+=20) {
            for (usz k = STENCIL_ORDER; k < dim_z - STENCIL_ORDER; k+=20) {

                // Cache blocking
                for (usz ii = 0; ii < 20; ++ii) {
                    for (usz jj = 0; jj < 20; ++jj) {
                        for (usz kk = 0; kk < 20; ++kk) {
                            tmp = A->cells[i+ii][j+jj][k+kk].value * B->cells[i+ii][j+jj][k+kk].value;

                            for (usz o = 1; o <= STENCIL_ORDER; ++o) {

                                // tmp = 1 / square_and_multiply(17.0, o);
                                // tmp = 1/pow(17.0, (f64)o);

                                tmp += A->cells[i+ii + o][j+jj][k+kk].value *
                                                        B->cells[i+ii + o][j+jj][k+kk].value * o_values[o-1];
                                tmp += A->cells[i+ii- o][j+jj][k+kk].value *
                                                        B->cells[i+ii - o][j+jj][k+kk].value * o_values[o-1];
                                tmp += A->cells[i+ii][j+jj + o][k+kk].value *
                                                        B->cells[i+ii][j+jj + o][k+kk].value * o_values[o-1];
                                tmp += A->cells[i+ii][j+jj - o][k+kk].value *
                                                        B->cells[i+ii][j+jj - o][k+kk].value * o_values[o-1];
                                tmp += A->cells[i+ii][j+jj][k+kk + o].value *
                                                        B->cells[i+ii][j+jj][k+kk + o].value * o_values[o-1];
                                tmp += A->cells[i+ii][j+jj][k+kk- o].value *
                                                        B->cells[i+ii][j+jj][k+kk - o].value * o_values[o-1];
                            }
                            C->cells[i+ii][j+jj][k+kk].value = tmp;
                        }
                    }
                }
            }
        }
    }

    mesh_copy_core(A, C);
}
