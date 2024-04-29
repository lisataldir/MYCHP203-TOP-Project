#include "stencil/solve.h"

#include <assert.h>
#include <math.h>

void solve_jacobi(mesh_t* A, mesh_t const* B, mesh_t* C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz const dim_x = A->dim_x;
    usz const dim_y = A->dim_y;
    usz const dim_z = A->dim_z;

    f64 tmp;

    for (usz i = STENCIL_ORDER; i < dim_x - STENCIL_ORDER; ++i) {
        for (usz j = STENCIL_ORDER; j < dim_y - STENCIL_ORDER; ++j) {
            for (usz k = STENCIL_ORDER; k < dim_z - STENCIL_ORDER; ++k) {
                C->cells[i][j][k].value = A->cells[i][j][k].value * B->cells[i][j][k].value;

                for (usz o = 1; o <= STENCIL_ORDER; ++o) {

                    tmp = 1/pow(17.0, (f64)o);

                    C->cells[i][j][k].value += A->cells[i + o][j][k].value *
                                               B->cells[i + o][j][k].value * tmp;
                    C->cells[i][j][k].value += A->cells[i - o][j][k].value *
                                               B->cells[i - o][j][k].value * tmp;
                    C->cells[i][j][k].value += A->cells[i][j + o][k].value *
                                               B->cells[i][j + o][k].value * tmp;
                    C->cells[i][j][k].value += A->cells[i][j - o][k].value *
                                               B->cells[i][j - o][k].value * tmp;
                    C->cells[i][j][k].value += A->cells[i][j][k + o].value *
                                               B->cells[i][j][k + o].value * tmp;
                    C->cells[i][j][k].value += A->cells[i][j][k - o].value *
                                               B->cells[i][j][k - o].value * tmp;
                }
            }
        }
    }

    mesh_copy_core(A, C);
}