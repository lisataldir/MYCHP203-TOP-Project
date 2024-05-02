#include "stencil/comm_handler.h"

#include "logging.h"

#include <stdio.h>
#include <unistd.h>

#define MAXLEN 8UL

static u32 gcd(u32 a, u32 b) {
    u32 c;
    while (b != 0) {
        c = a % b;
        a = b;
        b = c;
    }
    return a;
}

static char* stringify(char buf[static MAXLEN], i32 num) {
    snprintf(buf, MAXLEN, "%d", num);
    return buf;
}

comm_handler_t comm_handler_new(u32 rank, u32 comm_size, usz dim_x, usz dim_y, usz dim_z) {
    // Compute splitting
    u32 const nb_z = gcd(comm_size, (u32)(dim_x * dim_y));
    u32 const nb_y = gcd(comm_size / nb_z, (u32)dim_z);
    u32 const nb_x = (comm_size / nb_z) / nb_y;

    if (comm_size != nb_x * nb_y * nb_z) {
        error(
            "splitting does not match MPI communicator size\n -> expected %u, got %u",
            comm_size,
            nb_x * nb_y * nb_z
        );
    }

    // Compute current rank position
    u32 const rank_z = rank / (comm_size / nb_z);
    u32 const rank_y = (rank % (comm_size / nb_z)) / (comm_size / nb_y);
    u32 const rank_x = (rank % (comm_size / nb_z)) % (comm_size / nb_y);

    // Setup size
    usz const loc_dim_z = (rank_z == nb_z - 1) ? dim_z / nb_z + dim_z % nb_z : dim_z / nb_z;
    usz const loc_dim_y = (rank_y == nb_y - 1) ? dim_y / nb_y + dim_y % nb_y : dim_y / nb_y;
    usz const loc_dim_x = (rank_x == nb_x - 1) ? dim_x / nb_x + dim_x % nb_x : dim_x / nb_x;

    // Setup position
    u32 const coord_z = rank_z * (u32)dim_z / nb_z;
    u32 const coord_y = rank_y * (u32)dim_y / nb_y;
    u32 const coord_x = rank_x * (u32)dim_x / nb_x;

    // Compute neighboor nodes IDs
    i32 const id_left = (rank_x > 0) ? (i32)rank - 1 : -1;
    i32 const id_right = (rank_x < nb_x - 1) ? (i32)rank + 1 : -1;
    i32 const id_top = (rank_y > 0) ? (i32)(rank - nb_x) : -1;
    i32 const id_bottom = (rank_y < nb_y - 1) ? (i32)(rank + nb_x) : -1;
    i32 const id_front = (rank_z > 0) ? (i32)(rank - (comm_size / nb_z)) : -1;
    i32 const id_back = (rank_z < nb_z - 1) ? (i32)(rank + (comm_size / nb_z)) : -1;

    return (comm_handler_t){
        .nb_x = nb_x,
        .nb_y = nb_y,
        .nb_z = nb_z,
        .coord_x = coord_x,
        .coord_y = coord_y,
        .coord_z = coord_z,
        .loc_dim_x = loc_dim_x,
        .loc_dim_y = loc_dim_y,
        .loc_dim_z = loc_dim_z,
        .id_left = id_left,
        .id_right = id_right,
        .id_top = id_top,
        .id_bottom = id_bottom,
        .id_back = id_back,
        .id_front = id_front,
    };
}

void comm_handler_print(comm_handler_t const* self) {
    i32 rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    static char bt[MAXLEN];
    static char bb[MAXLEN];
    static char bl[MAXLEN];
    static char br[MAXLEN];
    static char bf[MAXLEN];
    static char bd[MAXLEN];
    fprintf(
        stderr,
        "****************************************\n"
        "RANK %d:\n"
        "  COORDS:     %u,%u,%u\n"
        "  LOCAL DIMS: %zu,%zu,%zu\n"
        "     %2s  %2s\n"
        "  %2s  \x1b[1m*\x1b[0m  %2s\n"
        "  %2s %2s\n",
        rank,
        self->coord_x,
        self->coord_y,
        self->coord_z,
        self->loc_dim_x,
        self->loc_dim_y,
        self->loc_dim_z,
        self->id_top < 0 ? " -" : stringify(bt, self->id_top),
        self->id_back < 0 ? " -" : stringify(bb, self->id_back),
        self->id_left < 0 ? " -" : stringify(bl, self->id_left),
        self->id_right < 0 ? " -" : stringify(br, self->id_right),
        self->id_front < 0 ? " -" : stringify(bf, self->id_front),
        self->id_bottom < 0 ? " -" : stringify(bd, self->id_bottom)
    );
}

void comm_handler_ghost_exchange(comm_handler_t const* self, mesh_t* mesh) {

    // Left to right phase
    if (self->id_right >= 0) MPI_Send(&(mesh->cells_value[(mesh->dim_x - 2*STENCIL_ORDER)*mesh->dim_y*mesh->dim_z]), mesh->dim_z*mesh->dim_y*STENCIL_ORDER, MPI_DOUBLE, self->id_right, 0, MPI_COMM_WORLD);
    if (self->id_left >= 0) MPI_Recv(mesh->cells_value, mesh->dim_z*mesh->dim_y*STENCIL_ORDER, MPI_DOUBLE, self->id_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Right to left phase
    if (self->id_left >= 0) MPI_Send(&(mesh->cells_value[STENCIL_ORDER*mesh->dim_y*mesh->dim_z]), mesh->dim_z*mesh->dim_y*STENCIL_ORDER, MPI_DOUBLE, self->id_left, 0, MPI_COMM_WORLD);
    if (self->id_right >= 0) MPI_Recv(&(mesh->cells_value[(mesh->dim_x - STENCIL_ORDER)*mesh->dim_y*mesh->dim_z]), mesh->dim_z*mesh->dim_y*STENCIL_ORDER, MPI_DOUBLE, self->id_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Prevent mixing communication from left/right with top/bottom and front/back
    MPI_Barrier(MPI_COMM_WORLD);

    // Top to bottom phase
    if (self->id_bottom >= 0) MPI_Send(&(mesh->cells_value[(mesh->dim_y - 2*STENCIL_ORDER)*mesh->dim_z]), mesh->dim_z*STENCIL_ORDER*mesh->dim_x, MPI_DOUBLE, self->id_bottom, 0, MPI_COMM_WORLD);
    if (self->id_top >= 0) MPI_Recv(mesh->cells_value, mesh->dim_z*STENCIL_ORDER*mesh->dim_x, MPI_DOUBLE, self->id_top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Bottom to top phase
    if (self->id_top >= 0) MPI_Send(&(mesh->cells_value[STENCIL_ORDER*mesh->dim_z]), mesh->dim_z*STENCIL_ORDER*mesh->dim_x, MPI_DOUBLE, self->id_top, 0, MPI_COMM_WORLD);
    if (self->id_bottom >= 0) MPI_Recv(&(mesh->cells_value[(mesh->dim_y - STENCIL_ORDER)*mesh->dim_z]), mesh->dim_z*STENCIL_ORDER*mesh->dim_x, MPI_DOUBLE, self->id_bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Prevent mixing communication from top/bottom with left/right and front/back
    MPI_Barrier(MPI_COMM_WORLD);

    // Front to back phase
    if (self->id_back >= 0) MPI_Send(&(mesh->cells_value[(mesh->dim_z - 2 * STENCIL_ORDER)]), STENCIL_ORDER * mesh->dim_x * mesh->dim_y, MPI_DOUBLE, self->id_back, 0, MPI_COMM_WORLD);
    if (self->id_front >= 0) MPI_Recv(mesh->cells_value, STENCIL_ORDER * mesh->dim_x * mesh->dim_y, MPI_DOUBLE, self->id_front, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Back to front phase
    if (self->id_front >= 0) MPI_Send(&(mesh->cells_value[STENCIL_ORDER]), STENCIL_ORDER * mesh->dim_x * mesh->dim_y, MPI_DOUBLE, self->id_front, 0, MPI_COMM_WORLD);
    if (self->id_back >= 0) MPI_Recv(&(mesh->cells_value[(mesh->dim_z - STENCIL_ORDER)]), STENCIL_ORDER * mesh->dim_x * mesh->dim_y, MPI_DOUBLE, self->id_back, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Need to synchronize all remaining in-flight communications before exiting
    MPI_Barrier(MPI_COMM_WORLD);
}


