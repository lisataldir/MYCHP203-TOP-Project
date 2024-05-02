#pragma once

#include "../types.h"

/// Three-dimensional mesh.
/// Storage of cells is in layout right (aka RowMajor).

#define STENCIL_ORDER 8UL

typedef struct mesh_s {
    f64* cells_value;
    i32* cells_kind; // cells_kind == 1 for cells core and 0 for cells phantoms
    usz dim_x;
    usz dim_y;
    usz dim_z;
    i32 mesh_kind; // mesh_kind == 0 if constant, 1 if intput and 2 if output
} mesh_t;

/// Initialize a mesh.
mesh_t mesh_new(usz dim_x, usz dim_y, usz dim_z, i32 mesh_kind);

/// De-initialize a mesh.
void mesh_drop(mesh_t* self);

/// Gets the kind of a cell in a mesh given its coordinates.
i32 mesh_set_cell_kind(mesh_t const* self, usz i, usz j, usz k);

/// Copies the inner part of a mesh into another.
void mesh_copy_core(mesh_t* dst, mesh_t const* src);
