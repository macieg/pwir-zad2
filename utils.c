#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

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