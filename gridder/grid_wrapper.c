#include "mex.h"
#include "matrix.h"
#include <math.h>

#include "grid.h"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int d, n, m;
    float *data_in;
    float *data_out;
    float *kt_pos;
    float *weight = NULL;
    unsigned long int *dims;
    int *fixed_dims;
    int ndim, npoints;
    int direction;
    long int num_matrix_elements = 1;

    gridding_parameters parm;
    point_vectors pv;
    
    double *r;
    double *i;
    int isComplex;
    
    /*----------------------------------------------------------------*/
    if(nrhs < 6) mexErrMsgTxt("Incorrect no. input arguments.");
    if(nlhs != 1) mexErrMsgTxt("Incorrect no. output arguments.");
    
    /* Arg0: kt_pos             */
    /* Arg1: data_in            */
    /* Arg2: dims               */
    /* Arg3: weight             */
    /* Arg4: direction          */
    /* Arg5: fixed_dims         */
    /* Arg6: Oversampling       */
    /* Arg7: Kernel Width       */
    /* Arg8: kernel Beta        */
    /*----------------------------------------------------------------*/

    /* Get and map some parameters */
    direction = (int)mxGetScalar(prhs[4]);
    ndim = mxGetN(prhs[0]);
    npoints = mxGetM(prhs[0]);
    
    if (ndim>3)
    {
        mexErrMsgTxt("Number of dimensions larger than 3. Not implemented yet.\n");
    }
    
    kt_pos = malloc(sizeof(float)*ndim*npoints);
    if (kt_pos == NULL)
    {
        mexErrMsgTxt("Unable to allocate kt_pos array\n");
    }
    
    r = mxGetPr(prhs[0]);
    for (d = 0; d<ndim;d++)
    {
        for (n=0; n<npoints; n++)
        {
            kt_pos[npoints*d+n] = (float)r[npoints*d+n];
        }
    }
    
    isComplex = mxIsComplex(prhs[1]);
    r = mxGetPr(prhs[1]);
    i = mxGetPi(prhs[1]);
    n = mxGetNumberOfElements(prhs[1]);
    data_in = malloc(sizeof(float)*n*2);
    if (data_in == NULL)
    {
        free(kt_pos);
        mexErrMsgTxt("Unable to allocate data_in array\n");
    }    
    for (d=0;d<n;d++)
    {
        data_in[d*2  ] = (float)r[d];
        if (isComplex) 
        {
            data_in[d*2+1] = (float)i[d];
        }
        else
        {
            data_in[d*2+1] = 0.0;        
        }
        
    }

    dims = malloc(sizeof(unsigned long)*ndim); //int
    if (dims == NULL)
    {
        free(kt_pos);
        free(data_in);
        mexErrMsgTxt("Unable to allocate dims array\n");
    }

    fixed_dims = malloc(sizeof(int)*ndim);
    if (fixed_dims == NULL)
    {
        free(kt_pos);
        free(data_in);
        free(dims);
        mexErrMsgTxt("Unable to allocate dims array\n");
    }
    
    init_gridding_parameters(&parm);
    
    if(nrhs > 6)
    {
        parm.over_sampling = mxGetScalar(prhs[6]);
    }
    
    if(nrhs > 7)
    {
        parm.kernel_width = mxGetScalar(prhs[7]);
    }
    
    if(nrhs > 8)
    {
        parm.kernel_beta = mxGetScalar(prhs[8]);
    }
    
    for (d=0;d<ndim;d++)
    {
        r = mxGetPr(prhs[2]);
        dims[d] = (unsigned long)r[d];
        i = mxGetPr(prhs[5]);
        fixed_dims[d] = (int)i[d];
        if (fixed_dims[d])
        {
            num_matrix_elements *= dims[d];
        }
        else
        {
            num_matrix_elements *= dims[d]*parm.over_sampling;
        }
    }
    
    calculate_point_vectors(kt_pos,npoints,ndim,dims,&parm,&pv,fixed_dims);
    calculate_kernel_tables(&parm);

    if (direction)
    {
        data_out = calloc(2*num_matrix_elements,sizeof(float));
        plhs[0] = mxCreateDoubleMatrix(num_matrix_elements,1,mxCOMPLEX);        
        n=num_matrix_elements;
    }
    else
    {
        data_out = calloc(2*npoints,sizeof(float));
        plhs[0] = mxCreateDoubleMatrix(npoints,1,mxCOMPLEX);    
        n=npoints;
    }
    
    if (data_out == NULL)
    {
        free(kt_pos);
        free(data_in);
        free(dims);
        free(fixed_dims);
        mexErrMsgTxt("Unable to allocate data_out array\n");
    }
    
    /* Do we have weights? */
    if (!mxIsEmpty(prhs[3]))
    {
        if (mxGetNumberOfElements(prhs[1]) == mxGetNumberOfElements(prhs[3]))
        {
            weight = malloc(sizeof(float)*npoints);
            if (weight == NULL )
            {
                free(kt_pos);
                free(data_in);
                free(dims);
                free(data_out);
                free(fixed_dims);
                mexErrMsgTxt("Unable to allocate weight array\n");
            }
            r = mxGetPr(prhs[3]);
            m = mxGetNumberOfElements(prhs[3]);
            for (d=0;d<m;d++)
            {
                weight[d] = r[d];
            }
        }
        else
        {
           free(kt_pos);
           free(data_in);
           free(dims);
           free(data_out);
           free(fixed_dims);
           mexErrMsgTxt("Size of data and weights do not match\n");
        }
    }
    
    if (ndim == 2)
    {
        genkt_grid_2d(&parm, pv.kernel_positions, pv.grid_positions, npoints, data_in, weight, data_out, dims, direction, fixed_dims);
    }
    else if (ndim == 3)
    {
        genkt_grid_3d(&parm, pv.kernel_positions, pv.grid_positions, npoints, data_in, weight, data_out, dims, direction, fixed_dims);
    }

    r = mxGetPr(plhs[0]);
    i = mxGetPi(plhs[0]);
    for(d=0;d<n;d++)
    {
        r[d] = data_out[d*2  ];
        i[d] = data_out[d*2+1];
    }
    
    for (d=0;d<ndim;d++)
    {
        if (!fixed_dims[d]) dims[d] *= parm.over_sampling;
    }

    /*if (direction) {
        if(mxSetDimensions(plhs[0], dims, ndim)) {
            mxDestroyArray(plhs[0]);
            mexErrMsgTxt("Failed to set the size of output");
        }
    }
    */
    
    /* clean up */
    free(kt_pos);
    free(data_in);
    free(dims);
    free(data_out);
    free(fixed_dims);
    delete_kernel_tables(&parm);
    delete_point_vectors(&pv);
    if (weight != NULL) free(weight);
}