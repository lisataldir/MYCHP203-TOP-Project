#include "stencil/solve.h"

#include <assert.h>
#include <stdio.h>
#include <math.h>

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

    // f64 o_values[STENCIL_ORDER]
    // for (usz o = 1; o <= STENCIL_ORDER; ++o) {
    //   o_values[o - 1] = 1 / square_and_multiply(17.0, o);
    // }

    f64 o_values[STENCIL_ORDER];
    for (usz o = 1; o <= STENCIL_ORDER; ++o)
    {
      o_values[o - 1] = 1 / square_and_multiply(17.0, o);
    }
    f64 tmp;
    f64 o_tmp;

    for (usz i = STENCIL_ORDER; i < dim_x - STENCIL_ORDER; ++i)  {
        for (usz j = STENCIL_ORDER; j < dim_y - STENCIL_ORDER; ++j) {
            for (usz k = STENCIL_ORDER; k < dim_z - STENCIL_ORDER; ++k){
                tmp = A->cells_value[i*dim_y*dim_z + j*dim_z + k] * B->cells_value[i*dim_y*dim_z + j*dim_z + k];

                for (usz o = 1; o <= STENCIL_ORDER; ++o) {
                    o_tmp = o_values[o-1];
                    tmp += A->cells_value[(i + o)*dim_y*dim_z + j*dim_z + k] *
                        B->cells_value[(i + o)*dim_y*dim_z + j*dim_z + k]  * o_tmp;
                    tmp += A->cells_value[(i - o)*dim_y*dim_z + j*dim_z + k] *
                        B->cells_value[(i - o)*dim_y*dim_z + j*dim_z + k] * o_tmp;
                    tmp += A->cells_value[i*dim_y*dim_z + (j + o)*dim_z + k] *
                        B->cells_value[i*dim_y*dim_z + (j + o)*dim_z + k] * o_tmp;
                    tmp += A->cells_value[i*dim_y*dim_z + (j - o)*dim_z + k] *
                        B->cells_value[i*dim_y*dim_z + (j - o)*dim_z + k] * o_tmp;
                    tmp += A->cells_value[i*dim_y*dim_z + j*dim_z + (k+o)] *
                        B->cells_value[i*dim_y*dim_z + j*dim_z + (k+o)]  * o_tmp;
                    tmp += A->cells_value[i*dim_y*dim_z + j*dim_z + (k-o)] *
                        B->cells_value[i*dim_y*dim_z + j*dim_z + (k-o)] * o_tmp;
                }

                C->cells_value[i*dim_y*dim_z + j*dim_z + k] = tmp;
            }
        }
    }

    mesh_copy_core(A, C);
}

// Cache blocking v1 (seg fault pour le moment)
// void solve_jacobi(mesh_t* A, mesh_t const* B, mesh_t* C) {
//   assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
//   assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
//   assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

//   usz const dim_x = A->dim_x;
//   usz const dim_y = A->dim_y;
//   usz const dim_z = A->dim_z;

//   // f64 o_values[STENCIL_ORDER]
//   // for (usz o = 1; o <= STENCIL_ORDER; ++o) {
//   //   o_values[o - 1] = 1 / square_and_multiply(17.0, o);
//   // }

//   f64 o_values[STENCIL_ORDER];
//   for (usz o = 1; o <= STENCIL_ORDER; ++o)
//   {
//     o_values[o - 1] = 1 / square_and_multiply(17.0, o);
//   }

//   f64 tmp;
//   usz block_x = 9;
//   usz block_y = 9;
//   usz block_z = 9;

//   for (usz i = STENCIL_ORDER; i < dim_x - STENCIL_ORDER; i+=block_x)
//   {
//     for (usz j = STENCIL_ORDER; j < dim_y - STENCIL_ORDER; j+=block_y) 
//     {
//       for (usz k = STENCIL_ORDER; k < dim_z - STENCIL_ORDER; k+=block_z) 
//       {

//         // Cache blocking
//         usz bi_end = (usz)fmin(i + block_x, dim_x - STENCIL_ORDER);
//         usz bj_end = (usz)fmin(j + block_y, dim_y - STENCIL_ORDER);
//         usz bk_end = (usz)fmin(k + block_z, dim_z - STENCIL_ORDER);

//         for (usz ii = i; ii < bi_end; ++ii) 
//         {
//           for (usz jj = j; jj < bj_end; ++jj) 
//           {
//             for (usz kk = k; kk < bk_end; ++kk) 
//             {

//               tmp = A->cells_value[(i+ii)*dim_y*dim_z + (j+jj)*dim_z + (k+kk)] * B->cells_value[(i+ii)*dim_y*dim_z + (j+jj)*dim_z + (k+kk)];

//               for (usz o = 1; o <= STENCIL_ORDER; ++o) {

//                 tmp += A->cells_value[(i+ii+o)*dim_y*dim_z + (j+jj)*dim_z + (k+kk)] *
//                        B->cells_value[(i+ii+o)*dim_y*dim_z + (j+jj)*dim_z + (k+kk)] * o_values[o-1];
//                 tmp += A->cells_value[(i+ii-o)*dim_y*dim_z + (j+jj)*dim_z + (k+kk)] *
//                        B->cells_value[(i+ii-o)*dim_y*dim_z + (j+jj)*dim_z + (k+kk)] * o_values[o-1];
//                 tmp += A->cells_value[(i+ii)*dim_y*dim_z + (j+jj+o)*dim_z + (k+kk)] *
//                        B->cells_value[(i+ii)*dim_y*dim_z + (j+jj+o)*dim_z + (k+kk)] * o_values[o-1];
//                 tmp += A->cells_value[(i+ii)*dim_y*dim_z + (j+jj-o)*dim_z + (k+kk)] *
//                        B->cells_value[(i+ii)*dim_y*dim_z + (j+jj-o)*dim_z + (k+kk)] * o_values[o-1];
//                 tmp += A->cells_value[(i+ii)*dim_y*dim_z + (j+jj)*dim_z + (k+kk+o)] *
//                        B->cells_value[(i+ii)*dim_y*dim_z + (j+jj)*dim_z + (k+kk+o)] * o_values[o-1];
//                 tmp += A->cells_value[(i+ii)*dim_y*dim_z + (j+jj)*dim_z + (k+kk-o)] *
//                        B->cells_value[(i+ii)*dim_y*dim_z + (j+jj)*dim_z + (k+kk-o)] * o_values[o-1];
//               }
//               C->cells_value[(i+ii)*dim_y*dim_z + (j+jj)*dim_z + (k+kk)] = tmp;
//             }
//           }
//         }
//       }
//     }
//   }

//   mesh_copy_core(A, C);
// }