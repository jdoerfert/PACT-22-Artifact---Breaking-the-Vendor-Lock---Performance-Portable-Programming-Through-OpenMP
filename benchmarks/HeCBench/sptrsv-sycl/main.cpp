#include "sptrsv.h"
#include "mmio.h"
#include "tranpose.h"
#include "read_mtx.h"

int main(int argc, char ** argv)
{
    if(argc != 2)
    {
        printf("Usage:/sptrsv example.mtx\n");
        return -1;
    }

    // load matrix data from file
    char  *filename;
    filename = argv[1];
    
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;
    
    read_mtx(filename, &m, &n, &nnzA, &csrRowPtrA, &csrColIdxA, &csrValA);
    
    // extract L with the unit-lower triangular sparsity structure of A
    int nnzL = 0;
    int *csrRowPtrL_tmp ;
    int *csrColIdxL_tmp ;
    VALUE_TYPE *csrValL_tmp;
    
    if(m<=n)
        n=m;
    else
        m=n;
    if (m<=1)
        return 0;
    
    change2tran(m, nnzA,csrRowPtrA, csrColIdxA, csrValA, &nnzL, &csrRowPtrL_tmp, &csrColIdxL_tmp, &csrValL_tmp);
    
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);
    
    if(m==0 || nnzL==0) return -3;
    
    
    int layer;
    double parallelism;
    
    matrix_layer(m,n,nnzL,csrRowPtrL_tmp,csrColIdxL_tmp,csrValL_tmp,&layer,&parallelism);
    
    VALUE_TYPE *x_ref;
    VALUE_TYPE *b ;
    get_x_b(m, n, csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, &x_ref, &b);
    VALUE_TYPE *x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m);
    
    sptrsv_syncfree(csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, m, n, nnzL, x, b, x_ref);
    
    free(csrColIdxL_tmp);
    free(csrValL_tmp);
    free(csrRowPtrL_tmp);
    free(x_ref);
    free(b);
    free(x);

    return 0;
}
