#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "densematgen.h"

//dla danego procesy zwraca liczbe kolumn w podzielonym A
int proc_to_cols_no(int mpi_rank, int num_processes, int cols_no) {
    int cpp = cols_no / num_processes; //podstawa
    int more = cols_no % num_processes; //liczba procesow ktorym dodamy 1 (pierwsze z brzegu)

    return cpp + ((mpi_rank < more) ? 1 : 0);
}

void send_sparse_async(MPI_Request *req, int desination, sparse_type *msg) {
    MPI_Isend(&(msg->cols_no), 1, MPI_INT, desination, COLS_NO_TAG, MPI_COMM_WORLD, req + 3);
    MPI_Isend(msg->IA, msg->rows_no + 1, MPI_INT, desination, IA_TAG, MPI_COMM_WORLD, req);
    MPI_Isend(msg->JA, msg->IA[msg->rows_no], MPI_INT, desination, JA_TAG, MPI_COMM_WORLD, req + 1);
    MPI_Isend(msg->A, msg->IA[msg->rows_no], MPI_DOUBLE, desination, A_TAG, MPI_COMM_WORLD, req + 2);
}

void receive_sparse_rows_no(int source, int rows_no, sparse_type *msg) {
    int nnz;
    msg->rows_no = rows_no;
    msg->IA = malloc((rows_no + 1) * sizeof(int)); //TODO free

    MPI_Recv(msg->IA, rows_no + 1, MPI_INT, source, IA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    nnz = msg->IA[rows_no];

    msg->JA = malloc(nnz * sizeof(int)); //TODO free
    msg->A = malloc(nnz * sizeof(double)); //TODO free

    MPI_Recv(msg->JA, nnz, MPI_INT, source, JA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(msg->A, nnz, MPI_DOUBLE, source, A_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&(msg->cols_no), 1, MPI_DOUBLE, source, COLS_NO_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void send_sparse(int desination, sparse_type *msg) {
    MPI_Send(&(msg->rows_no), 1, MPI_INT, desination, 0, MPI_COMM_WORLD);
    MPI_Send(&(msg->cols_no), 1, MPI_INT, desination, 0, MPI_COMM_WORLD);
    MPI_Send(msg->IA, msg->rows_no + 1, MPI_INT, desination, 0, MPI_COMM_WORLD);
    MPI_Send(msg->JA, msg->IA[msg->rows_no], MPI_INT, desination, 0, MPI_COMM_WORLD);
    MPI_Send(msg->A, msg->IA[msg->rows_no], MPI_DOUBLE, desination, 0, MPI_COMM_WORLD);
}

void receive_sparse(int source, sparse_type *msg) {
    int size;
    MPI_Status status;
    MPI_Recv(&(msg->rows_no), 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&(msg->cols_no), 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    msg->IA = malloc((msg->rows_no + 1) * sizeof(int));
    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_INT, &size);
    MPI_Recv(msg->IA, size, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    msg->JA = malloc((msg->IA[msg->rows_no]) * sizeof(int));
    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_INT, &size);
    MPI_Recv(msg->JA, size, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    msg->A = malloc((msg->IA[msg->rows_no]) * sizeof(double));
    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &size);
    MPI_Recv(msg->A, size, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

//dla danej kolumny macierzy czesciowej procesu zwraca jej indeks w macierzy globalnej
int col_part_to_col(int col_no, int mpi_rank, int num_processes, int cols_no) {
    int cpp = cols_no / num_processes; //podstawa
    int more = cols_no % num_processes; //liczba procesow ktorym dodamy 1 (pierwsze z brzegu)

    int procmorecols = more * (cpp+1); //liczba kolumn zabranych przez procesy z wieksza iloscia kolumn
    if (mpi_rank < more) { //indeks w kolumnie procesu z wieksza iloscia kolumn
        return mpi_rank * (cpp+1) + col_no;
    } else { //indeks w kolumnie procesu z mniejsza iloscia kolumn
//        if (mpi_rank == 1) {printf("procmorecols=%d + (mpi_rank=%d - more=%d) * cpp=%d + cols_no=%d\n", procmorecols, mpi_rank, more, cpp, cols_no);}
        return procmorecols + (mpi_rank - more) * cpp + col_no;
    }
}

void generate_dense(dense_type *dense, int rows_no, int cols_no, int mpi_rank, int num_processes, int gen_seed) {
    dense->cols_no = cols_no;
    dense->rows_no = rows_no;
    dense->vals = malloc(dense->rows_no * dense->cols_no * sizeof(double));
    int i, j;
    for (i = 0; i < dense->rows_no; ++i) {
        for (j = 0; j < dense->cols_no; ++j) {
            dense->vals[ARRAY_IND(i, j, dense->cols_no)] =
                    generate_double(gen_seed, i, col_part_to_col(j, mpi_rank, num_processes, rows_no)); //rows_no == global cols_no
        }
    }
}

void split_comm(int mpi_rank, int repl_fact, int num_processes, MPI_Comm *sub_comm, int *sub_rank, int *sub_size) {
    int color = mpi_rank / repl_fact;
    MPI_Comm_split(MPI_COMM_WORLD, color, num_processes, sub_comm);

    MPI_Comm_rank(*sub_comm, sub_rank);
    MPI_Comm_size(*sub_comm, sub_size);
}


void alloc_sparse(sparse_type *st, int nnz, int rows_no, int cols_no) {
    st->A = malloc(nnz * sizeof(double));
    st->IA = malloc((rows_no + 1) * sizeof(int));
    st->JA = malloc(nnz * sizeof(int));
    st->rows_no = rows_no;
    st->cols_no = cols_no;
}

void free_sparse(sparse_type *st) {
    free(st->A);
    free(st->IA);
    free(st->JA);
//    free(st); //TODO
}

void parse_csr(FILE *fd, sparse_type *st) {
    int nnz, rows_no, max_nnz_in_row, cols_no;
    fscanf(fd, "%d %d %d %d", &rows_no, &cols_no, &nnz, &max_nnz_in_row); //matrices are square, TODO max_nnz_in_row
    alloc_sparse(st, nnz, rows_no, cols_no);
    int i;
    for (i = 0; i < nnz; i++) {
        fscanf(fd, "%lf", st->A + i);
    }
    for (i = 0; i <= st->rows_no; i++) {
        fscanf(fd, "%d", st->IA + i);
    }
    for (i = 0; i < nnz; i++) {
        fscanf(fd, "%d", st->JA + i);
    }
}

void alloc_dense(int col_no, int rows_no, dense_type *m) {
    m->cols_no = col_no;
    m->rows_no = rows_no;
    m->vals = malloc(m->cols_no * m->rows_no * sizeof(double));
    int i;
    for (i = 0; i < m->cols_no * m->rows_no; ++i)
        m->vals[i] = 0;
}

void free_dense(dense_type *m) {
    free(m->vals);
}

void copy_dense(dense_type *source, dense_type *destination) {
    int i;
    destination->rows_no = source->rows_no;
    destination->cols_no = source->cols_no;
    for (i = 0; i < source->rows_no * source->cols_no; i++) {
        destination->vals[i] = source->vals[i];
    }
}

void DEBUG_ALL_SPARSES(int num_processes, sparse_type *msgs) {
    int k;
    for (k = 0; k < num_processes; k++) {
        DEBUG_SPARSE(k, msgs + k);
    }
}

int ROWS_NO(sparse_type* msg) {
    int i;
    int cols_no = 0;
    for (i = 0; i < msg->rows_no; i++) {
        if (msg->JA[i] > cols_no) {
            cols_no = msg->JA[i];
        }
    }
    return cols_no;
}

void DEBUG_SPARSE(int k, sparse_type *msg) {
    printf("==========================PROCES %d==========================\n", k);
    int i, j;
    double m[100][100];
    for (i = 0; i < 100; i++) for (j = 0; j < 100; j++) m[i][j] = 0;
    j = 0;
    while (j <= msg->rows_no) {
        for (i = msg->IA[j - 1]; i < msg->IA[j]; i++) {
            m[j - 1][msg->JA[i]] = msg->A[i];
        }
        j++;
    }

    for (i = 0; i < msg->rows_no; i++) {
        for (j = 0; j < msg->cols_no; j++) {
            printf("%.0lf ", m[i][j]);
        }
        printf("\n");
    }
}

void DEBUG_SPARSE_CSR(int k, sparse_type *msg) {
    printf("==========================PROCES %d==========================\n", k);
    printf("row_no=%d, col_no=%d\n", msg->rows_no, msg->cols_no);
    int i;
    for (i = 0; i < msg->IA[msg->rows_no]; i++) {
        printf("(A[%d]=%.0lf, JA[%d]=%d)\n", i, msg->A[i], i, msg->JA[i]);
    }
    for (i = 0; i < msg->rows_no + 1; i++){
        printf("IA[%d]=%d\n", i, msg->IA[i]);
    }
}

void DEBUG_DENSE(int mpi_rank, dense_type *d) {
    int i, j;
    i = j = 0;
    printf("==========================PROCES %d==========================\n", mpi_rank);
    for (i = 0; i < d->rows_no; i++) {
        for (j = 0; j < d->cols_no; j++) {
            printf("%.0lf ", d->vals[i * d->cols_no + j]);
        }
        printf("\n");
    }
}

void DEBUG_SPARSE_TO_DENSE(int k, int num_processes, int sub_size, sparse_type *msg) {
    printf("==========================PROCES %d==========================\n", k);
    int i, j;
    double m[100][100];
    for (i = 0; i < 100; i++) for (j = 0; j < 100; j++) m[i][j] = 0;
    j = 0;
    while (j <= msg->rows_no) {
        for (i = msg->IA[j - 1]; i < msg->IA[j]; i++) {
            m[j - 1][msg->JA[i]] = msg->A[i];
        }
        j++;
    }

    for (i = 0; i < msg->rows_no; i++) {
        for (j = 0; j < ROWS_NO(msg); j++) {
            printf("%.5lf ", m[i][j]);
        }
        printf("\n");
    }
}