#include "stencil/mesh.h"

#include "logging.h"

#include <assert.h>
#include <stdlib.h>

mesh_t mesh_new(usz dim_x, usz dim_y, usz dim_z, i32 mesh_kind) {
    usz const ghost_size = 2 * STENCIL_ORDER;

    f64* cells_value = malloc((dim_z + ghost_size)*(dim_y + ghost_size)*(dim_x + ghost_size) * sizeof(f64));
    if (NULL == cells_value) {
        error("failed to allocate mesh cells of size %zu bytes", (dim_x + ghost_size)*(dim_y + ghost_size)*(dim_z + ghost_size));
    }
    i32* cells_kind = malloc((dim_z + ghost_size)*(dim_y + ghost_size)*(dim_x + ghost_size)*sizeof(i32));
    if (NULL == cells_kind) {
        error("failed to allocate mesh cells kinds of size %zu bytes", (dim_x + ghost_size)*(dim_y + ghost_size)*(dim_z + ghost_size));
    }

    return (mesh_t){
        .cells_value = cells_value,
        .cells_kind = cells_kind,
        .dim_x = dim_x + ghost_size,
        .dim_y = dim_y + ghost_size,
        .dim_z = dim_z + ghost_size,
        .mesh_kind = mesh_kind,
    };
}

void mesh_drop(mesh_t* self) {
    if (NULL != self->cells_value) {
        free(self->cells_value);
    }
    if (NULL != self->cells_kind) {
        free(self->cells_kind);
    }
}

i32 mesh_set_cell_kind(mesh_t const* self, usz i, usz j, usz k) {
    if ((i >= STENCIL_ORDER && i < self->dim_x - STENCIL_ORDER) &&
        (j >= STENCIL_ORDER && j < self->dim_y - STENCIL_ORDER) &&
        (k >= STENCIL_ORDER && k < self->dim_z - STENCIL_ORDER))
    {
        return 1; // cores cells
    } else {
        return 0; // ghosts cells
    }
}

void mesh_copy_core(mesh_t* dst, mesh_t const* src) {
    assert(dst->dim_x == src->dim_x);
    assert(dst->dim_y == src->dim_y);
    assert(dst->dim_z == src->dim_z);

    for (usz i = STENCIL_ORDER; i < dst->dim_x - STENCIL_ORDER; ++i) {
        for (usz j = STENCIL_ORDER; j < dst->dim_y - STENCIL_ORDER; ++j) {
            for (usz k = STENCIL_ORDER; k < dst->dim_z - STENCIL_ORDER; ++k) {
                assert(dst->cells_kind[i*dst->dim_y*dst->dim_z + j*dst->dim_z + k] == 1);
                assert(src->cells_kind[i*src->dim_y*src->dim_z + j*src->dim_z + k] == 1);
                dst->cells_value[i*dst->dim_y*dst->dim_z + j*dst->dim_z + k] = 
                src->cells_value[i*src->dim_y*src->dim_z + j*src->dim_z + k];
            }
        }
    }
}