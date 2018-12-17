import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport dgemm
from scipy.linalg.cython_blas cimport sgemm
from scipy.linalg.cython_blas cimport cgemm
from scipy.linalg.cython_blas cimport zgemm
from cython cimport boundscheck
from cython.parallel import prange, parallel


@boundscheck(False)
cpdef int fdgemm(double[:,::1] a, double[:,::1] b, double[:,::1] c, double alpha=1.0, double beta=0.0,
                char *transa='n', char *transb='n') nogil except -1:

    cdef:
        int m, n, k, lda, ldb, ldc
        double *a0=&a[0,0]
        double *b0=&b[0,0]
        double *c0=&c[0,0]

    ldb = (&a[1,0]) - a0 if a.shape[0] > 1 else 1
    lda = (&b[1,0]) - b0 if b.shape[0] > 1 else 1

    k = b.shape[0]
    if k != a.shape[1]:
        with gil:
            raise ValueError("Shape mismatch in input arrays.")
    m = b.shape[1]
    n = a.shape[0]
    if n != c.shape[0] or m != c.shape[1]:
        with gil:
            raise ValueError("Output array does not have the correct shape.")
    ldc = (&c[1,0]) - c0 if c.shape[0] > 1 else 1
    dgemm(transa, transb, &m, &n, &k, &alpha, b0, &lda, a0,
          &ldb, &beta, c0, &ldc)
    return 0


@boundscheck(False)
cpdef int fsgemm(float[:,::1] a, float[:,::1] b, float[:,::1] c, float alpha=1.0, float beta=0.0,
                char *transa='n', char *transb='n') nogil except -1:

    cdef:
        int m, n, k, lda, ldb, ldc
        float *a0=&a[0,0]
        float *b0=&b[0,0]
        float *c0=&c[0,0]

    ldb = (&a[1,0]) - a0 if a.shape[0] > 1 else 1
    lda = (&b[1,0]) - b0 if b.shape[0] > 1 else 1

    k = b.shape[0]
    if k != a.shape[1]:
        with gil:
            raise ValueError("Shape mismatch in input arrays.")
    m = b.shape[1]
    n = a.shape[0]
    if n != c.shape[0] or m != c.shape[1]:
        with gil:
            raise ValueError("Output array does not have the correct shape.")
    ldc = (&c[1,0]) - c0 if c.shape[0] > 1 else 1
    sgemm(transa, transb, &m, &n, &k, &alpha, b0, &lda, a0,
          &ldb, &beta, c0, &ldc)
    return 0


@boundscheck(False)
cpdef int fcgemm(float complex[:,::1] a, float complex[:,::1] b, float complex[:,::1] c, float complex alpha=1.0, float complex beta=0.0,
                char *transa='n', char *transb='n') nogil except -1:

    cdef:
        int m, n, k, lda, ldb, ldc
        float complex *a0=&a[0,0]
        float complex *b0=&b[0,0]
        float complex *c0=&c[0,0]

    ldb = (&a[1,0]) - a0 if a.shape[0] > 1 else 1
    lda = (&b[1,0]) - b0 if b.shape[0] > 1 else 1

    k = b.shape[0]
    if k != a.shape[1]:
        with gil:
            raise ValueError("Shape mismatch in input arrays.")
    m = b.shape[1]
    n = a.shape[0]
    if n != c.shape[0] or m != c.shape[1]:
        with gil:
            raise ValueError("Output array does not have the correct shape.")
    ldc = (&c[1,0]) - c0 if c.shape[0] > 1 else 1
    cgemm(transa, transb, &m, &n, &k, &alpha, b0, &lda, a0,
          &ldb, &beta, c0, &ldc)
    return 0


@boundscheck(False)
cpdef int batch_zgemm(complex[:, :,::1] a, complex[:, :,::1] b, complex[:, :,::1] c,
                      complex alpha=1.0, complex beta=0.0, char *transa='n', char *transb='n') nogil except -1:

    cdef:
        int i, m, n, k, lda, ldb, ldc, bs
    bs = a.shape[0]

    if bs != b.shape[0]:
        with gil:
            raise ValueError("Shape mismatch in input arrays: a={}, b={}".format(a.shape, b.shape))
    if bs != c.shape[0]:
        with gil:
            raise ValueError("Shape mismatch in input arrays: a={}, b={}".format(c.shape, b.shape))
    k = b.shape[1]
    if k != a.shape[2]:
        with gil:
            raise ValueError("Shape mismatch in input arrays: a={}, b={}".format(a.shape, b.shape))
    m = b.shape[2]
    n = a.shape[1]
    if n != c.shape[1] or m != c.shape[2]:
        with gil:
            raise ValueError("Output array does not have the correct shape.")

    ldb = (&a[0,1,0]) - (&a[0,0,0]) if a.shape[1] > 1 else 1
    lda = (&b[0,1,0]) - (&b[0,0,0]) if b.shape[1] > 1 else 1
    ldc = (&c[0,1,0]) - (&c[0,0,0]) if c.shape[1] > 1 else 1
    for i in prange(bs, schedule='guided'):
        zgemm(transa, transb, &m, &n, &k, &alpha, &b[i,0,0], &lda, &a[i,0,0], &ldb, &beta, &c[i,0,0], &ldc)
    return 0


@boundscheck(False)
cpdef int batch_cgemm(float complex[:, :,::1] a, float complex[:, :,::1] b, float complex[:, :,::1] c,
                      float complex alpha=1.0, float complex beta=0.0, char *transa='n', char *transb='n') nogil except -1:

    cdef:
        int i, m, n, k, lda, ldb, ldc, bs
    bs = a.shape[0]

    # if bs != b.shape[0]:
    #     with gil:
    #         raise ValueError("Shape mismatch in input arrays: a={}, b={}".format(a.shape, b.shape))
    # if bs != c.shape[0]:
    #     with gil:
    #         raise ValueError("Shape mismatch in input arrays: a={}, b={}".format(c.shape, b.shape))
    k = b.shape[1]
    # if k != a.shape[2]:
    #     with gil:
    #         raise ValueError("Shape mismatch in input arrays: a={}, b={}".format(a.shape, b.shape))
    m = b.shape[2]
    n = a.shape[1]
    # if n != c.shape[1] or m != c.shape[2]:
    #     with gil:
    #         raise ValueError("Output array does not have the correct shape.")

    ldb = (&a[0,1,0]) - (&a[0,0,0]) if a.shape[1] > 1 else 1
    lda = (&b[0,1,0]) - (&b[0,0,0]) if b.shape[1] > 1 else 1
    ldc = (&c[0,1,0]) - (&c[0,0,0]) if c.shape[1] > 1 else 1
    for i in prange(bs, schedule='guided'):
        cgemm(transa, transb, &m, &n, &k, &alpha, &b[i,0,0], &lda, &a[i,0,0], &ldb, &beta, &c[i,0,0], &ldc)
    return 0
