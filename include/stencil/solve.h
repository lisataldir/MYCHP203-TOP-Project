#pragma once

#include "mesh.h"

void solve_jacobi(mesh_t *restrict A, mesh_t const *restrict B, mesh_t *restrict C);
