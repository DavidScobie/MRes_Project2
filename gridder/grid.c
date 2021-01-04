#include <stdlib.h> /* malloc */
#include <stdio.h>  /* printf */
#include <math.h>	/* floor, ceil */

#include "grid.h"

#ifndef SQR
#define SQR(x)		((x)*(x))
#endif

void init_gridding_parameters(gridding_parameters* parm)
{
    parm->over_sampling = GRID_DEFAULT_OVERSAMPLE_FACTOR;
    parm->kernel_width = GRID_DEFAULT_KERNEL_WIDTH;
    parm->kernel_table_steps = GRID_DEFAULT_KERNEL_TABLE_STEPS;
#ifndef GRID_KERNEL_GAUSSIAN        
    parm->kernel_beta = (float)GRID_DEFAULT_KERNEL_BETA;
#else
    parm->kernel_beta = (float)GRID_DEFAULT_KERNEL_SIGMA;
#endif
    parm->kernel_table = NULL;
    parm->transformed_kernel_table = NULL;
}


int calculate_kernel_tables(gridding_parameters* parm)
{
	int i;
	float k, k2;
    float kernel_norm;
    
	parm->kernel_samples = (int)parm->kernel_width*parm->kernel_table_steps+1;
	parm->kernel_table = (float *)malloc(sizeof(float)*parm->kernel_samples);
	if (parm->kernel_table == NULL)
	{
		printf("Error allocating kernel table!\n");
		return -1;
	}

    k=(float)(-parm->kernel_samples>>1);
    kernel_norm = 0;

#ifndef GRID_KERNEL_GAUSSIAN    
	for (i=0;i<parm->kernel_samples;i++)
	{
		k2 = 1.0f-SQR(2.0f*k/parm->kernel_samples);
		if (k2<0) k2=0; else k2=(float)sqrt(k2);  
		k2 = (float)bessi0(parm->kernel_beta * k2);
        parm->kernel_table[i] = k2;
        kernel_norm += k2;
        k++;
        /* mexPrintf("kernel_table[%d] = %f, %f\n", i, parm->kernel_table[i], k); */
	}
#else
    for (i=0;i<parm->kernel_samples;i++)
	{
        k2 = abs((1.0f*i-(parm->kernel_samples>>1)))*(parm->kernel_width/parm->kernel_samples);
		k2 = (float)exp(-0.5*(k2/parm->kernel_beta)*(k2/parm->kernel_beta));
        parm->kernel_table[i] = k2;
        kernel_norm += k2;
        k++;
        /* mexPrintf("kernel_table[%d] = %f, %f\n", i, parm->kernel_table[i], k); */
	}
#endif    
    
	for (i=0;i<parm->kernel_samples;i++)
	{
        parm->kernel_table[i] /= (kernel_norm/parm->kernel_table_steps);
	}
	return 0;
}

void delete_kernel_tables(gridding_parameters* parm)
{
    if (parm->kernel_table != NULL)
    {
    	free(parm->kernel_table);
        parm->kernel_table = NULL;
    }

    if (parm->transformed_kernel_table != NULL)
    {
    	free(parm->transformed_kernel_table);
        parm->transformed_kernel_table = NULL;
    }
}


int calculate_point_vectors(float* kt_pos, int npoints, int ndim, unsigned long* dims, gridding_parameters* parm, point_vectors* pv, int* fixed_dims)
{
	int i,j;
    float tmp;
    
	pv->number_of_points = npoints;
	pv->number_of_dimensions = ndim;
	pv->kernel_positions = NULL;
	pv->grid_positions = NULL;

	pv->kernel_positions = (int*) malloc(sizeof(int)*npoints*ndim);
	if (pv->kernel_positions == 0)
	{
		printf("calculate_point_vectors: Error allocating memory for integer kernel positions\n");
		return -1;
	}

	pv->grid_positions = (int*) malloc(sizeof(int)*npoints*ndim);
	if (pv->grid_positions == 0)
	{
		printf("calculate_point_vectors: Error allocating memory for integer grid positions\n");
		free(pv->kernel_positions);
		pv->kernel_positions = NULL;
		return -1;
	}

	for (i=0;i<npoints;i++)
	{
		for(j=0;j<ndim;j++)
		{
			if (fixed_dims[j])
			{
				pv->grid_positions[npoints*j+i] = (int)(kt_pos[npoints*j+i])+(dims[j]>>1);
				pv->kernel_positions[npoints*j+i] = (int)fabs((kt_pos[npoints*j+i]+(dims[j]>>1) - parm->kernel_width/2.0)*parm->kernel_table_steps - pv->grid_positions[npoints*j+i]*parm->kernel_table_steps);
			}
			else
			{
                tmp = kt_pos[npoints*j+i]+(dims[j]>>1) - parm->kernel_width/2.0;
				pv->grid_positions[npoints*j+i] = (int)ceil(tmp*parm->over_sampling);
                /* pv->grid_positions[npoints*j+i] = (int)floor(tmp*parm->over_sampling); */
				pv->kernel_positions[npoints*j+i] = (int)fabsf((tmp*parm->over_sampling - pv->grid_positions[npoints*j+i])*(parm->kernel_table_steps/parm->over_sampling));
			}
		}
	}

	return 0;
}

void delete_point_vectors(point_vectors* pv)
{
	if (pv->kernel_positions != NULL)
	{
		free(pv->kernel_positions);
		pv->kernel_positions = NULL;
	}

	if (pv->grid_positions != NULL)
	{
		free(pv->grid_positions);
		pv->grid_positions = NULL;
	}
}

int genkt_grid_2d(	gridding_parameters* parm, int *k_pos, int* grid_pos, int npoints, float *data_in, 
					float *weight, float *data_out, unsigned long *dims, int direction, int *fixed_dims)
{
    /* fixed_dims holds '1' if dimension should be skipped during gridding and zero if 0 */
    
	int n,x,y;
	float *weighted_data_in;
	float kx,ky;
	int o1,o;
	int kernel_limits[2];
	int kernel_step[2];
	int oversampled_dims[2];
	int ndim = 2;


	/*-----------------*/
	/* Apply weights   */
	/*-----------------*/
	if (weight != NULL && direction) /* Weights only make sense going onto the cartesian grid */
	{
		weighted_data_in = malloc(sizeof(float)*2*npoints);
		if (weighted_data_in == NULL)
		{
			printf("genkt_grid: Unable to allocate weighted_data_in array\n");
			return -1;
		}

		for (n=0;n<npoints;n++)
		{
			weighted_data_in[n*2  ] = data_in[n*2  ]*weight[n];   /* REAL */
			weighted_data_in[n*2+1] = data_in[n*2+1]*weight[n];   /* IMAG */
		}
	}
	else
	{
		weighted_data_in = data_in;
	}

	for (n=0;n<ndim;n++)
	{
		if (fixed_dims[n])
		{
			kernel_limits[n] = 1;
			oversampled_dims[n] = dims[n];
			kernel_step[n] = parm->kernel_table_steps;
		}
		else
		{
			kernel_limits[n] = parm->kernel_width*parm->over_sampling;
			oversampled_dims[n] = parm->over_sampling*dims[n];
			kernel_step[n] = parm->kernel_table_steps/parm->over_sampling;
		}
	}

	for (n=0;n<npoints;n++)
	{
		for (y=0;y<kernel_limits[1];y++)
		{
			if (fixed_dims[1])
			{
				ky = 1;
			}
			else
			{
				ky = parm->kernel_table[k_pos[npoints + n]+y*kernel_step[1]];
			}
			o1 = oversampled_dims[0]*((grid_pos[npoints + n]+y+oversampled_dims[1])%oversampled_dims[1]);
			for (x=0;x<kernel_limits[0];x++)
			{
				if (fixed_dims[0])
				{
					kx = 1;
				}
				else
				{
					kx = ky*(parm->kernel_table[k_pos[n]+x*kernel_step[0]]);
				}
				o = (o1 + (grid_pos[n]+x+oversampled_dims[0])%oversampled_dims[0])*2;
				if (direction)
				{
					data_out[o  ] += kx*weighted_data_in[n*2  ];
					data_out[o+1] += kx*weighted_data_in[n*2+1];
				}
				else
				{
					data_out[n*2  ] += kx*data_in[o  ];
					data_out[n*2+1] += kx*data_in[o+1];
				}
			}
		}
	}

	/*----------------*/
	/* Clean up       */
	/*----------------*/
	if (weight != NULL && direction)
	{
		free(weighted_data_in);
	}

	return 0;
}

int genkt_grid_3d(	gridding_parameters* parm, int *k_pos, int* grid_pos, int npoints, float *data_in, 
					float *weight, float *data_out, unsigned long *dims, int direction, int *fixed_dims)
{
    /* fixed_dims holds '1' if dimension should be skipped during gridding and zero if 0 */
    
	int n,x,y,z;
	float *weighted_data_in;
	float kx,ky,kz;
	int o2,o1,o;
	int kernel_limits[3];
	int kernel_step[3];
	int oversampled_dims[3];
	int ndim = 3;


	/*-----------------*/
	/* Apply weights   */
	/*-----------------*/
	if (weight != NULL && direction) /* Weights only make sense going onto the cartesian grid */
	{
		weighted_data_in = malloc(sizeof(float)*2*npoints);
		if (weighted_data_in == NULL)
		{
			printf("genkt_grid: Unable to allocate weighted_data_in array\n");
			return -1;
		}

		for (n=0;n<npoints;n++)
		{
			weighted_data_in[n*2  ] = data_in[n*2  ]*weight[n];   /* REAL */
			weighted_data_in[n*2+1] = data_in[n*2+1]*weight[n];   /* IMAG */
		}
	}
	else
	{
		weighted_data_in = data_in;
	}

	for (n=0;n<ndim;n++)
	{
		if (fixed_dims[n])
		{
			kernel_limits[n] = 1;
			oversampled_dims[n] = dims[n];
			kernel_step[n] = parm->kernel_table_steps;
		}
		else
		{
			kernel_limits[n] = parm->kernel_width*parm->over_sampling;
			oversampled_dims[n] = parm->over_sampling*dims[n];
			kernel_step[n] = parm->kernel_table_steps/parm->over_sampling;
		}
	}

	for (n=0;n<npoints;n++)
	{
        for (z=0;z<kernel_limits[2];z++)
        {
            if (fixed_dims[2])
            {
                kz = 1;
            }
            else
            {
                kz = parm->kernel_table[k_pos[npoints*2 + n]+z*kernel_step[2]];
            }
            o2 = oversampled_dims[1]*oversampled_dims[0]*((grid_pos[npoints*2 + n]+z+oversampled_dims[2])%oversampled_dims[2]);
            for (y=0;y<kernel_limits[1];y++)
            {
                if (fixed_dims[1])
                {
                    ky = 1;
                }
                else
                {
                    ky = kz * parm->kernel_table[k_pos[npoints + n]+y*kernel_step[1]];
                }
                o1 = o2 + oversampled_dims[0]*((grid_pos[npoints + n]+y+oversampled_dims[1])%oversampled_dims[1]);
                for (x=0;x<kernel_limits[0];x++)
                {
                    if (fixed_dims[0])
                    {
                        kx = 1;
                    }
                    else
                    {
                        kx = ky*(parm->kernel_table[k_pos[n]+x*kernel_step[0]]);
                    }
                    o = (o1 + (grid_pos[n]+x+oversampled_dims[0])%oversampled_dims[0])*2;
                    if (direction)
                    {
                        data_out[o  ] += kx*weighted_data_in[n*2  ];
                        data_out[o+1] += kx*weighted_data_in[n*2+1];
                    }
                    else
                    {
                        data_out[n*2  ] += kx*data_in[o  ];
                        data_out[n*2+1] += kx*data_in[o+1];
                    }
                }
            }
        }
	}

	/*----------------*/
	/* Clean up       */
	/*----------------*/
	if (weight != NULL && direction)
	{
		free(weighted_data_in);
	}

	return 0;
}


int genkt_grid(float *kt_pos, int npoints, float *data_in, float *weight, float *data_out, int ndim, int *dims, int *oversample_factors, float kernel_width, float kernel_beta, int direction, int *fixed_dims)
{

    /* fixed_dims holds '1' if dimension should be skipped during gridding and zero if 0 */
    
	int n, d;
	int *grid_values;
	int *sub_matrix_sizes;
	float *kernel_values;
	float *weighted_data_in;
	int *matrix_centers;
	double k;
	double kernel_norm;

	/*----------------*/
	/* Setup memory   */
	/*----------------*/
	grid_values = malloc(sizeof(int)*ndim*5);  /* Upper and lower limits for grid kernel, and the current point in the grid */
	if (grid_values == NULL)
	{
		printf("genkt_grid: Unable to allocate grid_values array\n");
		return -1;
	}


	kernel_values = malloc(sizeof(float)*ndim*2); /* Values of grid kernel along each dimension for current point and multiplication */
	if (kernel_values == NULL)
	{
		printf("genkt_grid: Unable to allocate kernel_values array\n");
		free(grid_values);
		return -1;
	}


	matrix_centers = malloc(sizeof(int)*ndim);  /* Upper and lower limits for grid kernel, and the current point in the grid */
	if (grid_values == NULL)
	{
		printf("genkt_grid: Unable to allocate matrix_centers array\n");
		free(grid_values);
		free(kernel_values);
		return -1;
	}

	sub_matrix_sizes = malloc(sizeof(int)*ndim);  /* Upper and lower limits for grid kernel, and the current point in the grid */
	if (grid_values == NULL)
	{
		printf("genkt_grid: Unable to allocate sub_matrix_sizes array\n");
		free(grid_values);
		free(kernel_values);
		free(matrix_centers);
		return -1;
	}

	sub_matrix_sizes[0] = 1;
	for (d=1;d<ndim; d++)
	{
		sub_matrix_sizes[d] = sub_matrix_sizes[d-1] * dims[d-1];
	}
	/*-----------------*/


	/*-----------------*/
	/* Apply weights   */
	/*-----------------*/
	if (weight != NULL && direction) /* Weights only make sense going onto the cartesian grid */
	{
		weighted_data_in = malloc(sizeof(float)*2*npoints);
		if (weighted_data_in == NULL)
		{
			printf("genkt_grid: Unable to allocate weighted_data_in array\n");
			free(grid_values);
			free(kernel_values);
			free(matrix_centers);
			free(sub_matrix_sizes);
			return -1;
		}

		for (n=0;n<npoints;n++)
		{
			weighted_data_in[n*2  ] = data_in[n*2  ]*weight[n];   /* REAL */
			weighted_data_in[n*2+1] = data_in[n*2+1]*weight[n];   /* IMAG */
		}
	}
	else
	{
		weighted_data_in = data_in;
	}


	/* Calculate Matrix Centers */
	for (d=0; d<ndim; d++) matrix_centers[d]=dims[d]>>1;


	/* Calculate kernel norm */
	kernel_norm = pow(1/bessi0(kernel_beta),ndim);
	for (n=0; n<npoints; n++)
	{ 
		/* Calculate limits for the kernel for the current point */
		for (d=0; d<ndim; d++)
		{
		    if (fixed_dims[d])
		    {
			    grid_values[d] = (int)((kt_pos[npoints*d+n])*oversample_factors[d]);
			    grid_values[ndim+d] = (int)((kt_pos[npoints*d+n])*oversample_factors[d]);		    
		    }
		    else
		    {
			    grid_values[d] = (int)ceil((kt_pos[npoints*d+n]-kernel_width/2)*oversample_factors[d]);
			    grid_values[ndim+d] = (int)floor((kt_pos[npoints*d+n]+kernel_width/2)*oversample_factors[d]);
			}
			grid_values[ndim*2+d] = grid_values[d]; /* start at lower limit */
			grid_values[ndim*3+d] = (grid_values[ndim*2+d]+matrix_centers[d]+dims[d])%dims[d]; /* Array, position, Circular wrapping */
			if (d == 0)
			{
				grid_values[ndim*4+d] = sub_matrix_sizes[d]*grid_values[ndim*3+d];
			}
			else
			{
				grid_values[ndim*4+d] = sub_matrix_sizes[d]*grid_values[ndim*3+d] + grid_values[ndim*4+d-1];
			}

			/* Update kernel values */
			k = ((double)grid_values[ndim*2+d])/((double)oversample_factors[d]) - kt_pos[npoints*d+n];
			k = 1.0-SQR(2.0*k/kernel_width);
			if (k<=0) k=0; else k=sqrt(k);    /* Prevent round off error below 0 */
			k = bessi0(kernel_beta * k);
			kernel_values[d] = (float)k;
			if (d == 0) 
			{
				kernel_values[ndim+d] = (float)(kernel_values[d]*kernel_norm);
			}
			else
			{
				kernel_values[ndim+d] = kernel_values[ndim+d-1] * kernel_values[d];
			}
		}

		while (grid_values[ndim*2] <= grid_values[ndim])
		{

			if (direction)
			{
				data_out[grid_values[ndim*5-1]*2  ] += weighted_data_in[n*2  ]*kernel_values[ndim*2-1]; /* REAL */
				data_out[grid_values[ndim*5-1]*2+1] += weighted_data_in[n*2+1]*kernel_values[ndim*2-1]; /* IMAG */
			}
			else
			{
				data_out[n*2  ] += data_in[grid_values[ndim*5-1]*2  ]*kernel_values[ndim*2-1]; /* REAL */
				data_out[n*2+1] += data_in[grid_values[ndim*5-1]*2+1]*kernel_values[ndim*2-1]; /* IMAG */
			}


			/* update grid value */
			grid_values[ndim*3-1]++;
			if (grid_values[ndim*3-1] > grid_values[ndim*2-1])
			{
				for (d=ndim-1; d>0; d--)
				{
					grid_values[ndim*2+d] = grid_values[d];
					grid_values[ndim*2+d-1]++;
					if (grid_values[ndim*2+d-1] <= grid_values[ndim+d-1]) break;
				}
				/* Recalculate kernel values */
				if (d==0) break;
				for (d=d-1 ; d<ndim; d++)
				{
					k = ((double)grid_values[ndim*2+d])/((double)oversample_factors[d]) - kt_pos[npoints*d+n];
					k = 1.0-SQR(2.0*k/kernel_width);
					if (k<=0) k=0; else k=sqrt(k);    /* Prevent round off error below 0 */
					k = bessi0(kernel_beta * k);
					kernel_values[d] = (float)k;
					if (d == 0) 
					{
						kernel_values[ndim+d] = (float)(kernel_values[d]*kernel_norm);
					}
					else
					{
						kernel_values[ndim+d] = kernel_values[ndim+d-1] * kernel_values[d];
					}

					grid_values[ndim*3+d] = (grid_values[ndim*2+d]+matrix_centers[d]+dims[d])%dims[d]; /* Array, position, Circular wrapping */
					if (d == 0)
					{
						grid_values[ndim*4+d] = sub_matrix_sizes[d]*grid_values[ndim*3+d];
					}
					else
					{
						grid_values[ndim*4+d] = sub_matrix_sizes[d]*grid_values[ndim*3+d] + grid_values[ndim*4+d-1];
					}

				}
			}
			else
			{
				k = ((double)grid_values[ndim*3-1])/((double)oversample_factors[ndim-1]) - kt_pos[npoints*(ndim-1)+n];
				k = 1.0-SQR(2.0*k/kernel_width);
				if (k<=0) k=0; else k=sqrt(k);    /* Prevent round off error below 0 */
				k = bessi0(kernel_beta * k);
				kernel_values[ndim-1] = (float)k;
				kernel_values[ndim*2-1] = kernel_values[ndim*2-2] * kernel_values[ndim-1];
				grid_values[ndim*4-1] = (grid_values[ndim*3-1]+matrix_centers[ndim-1]+dims[ndim-1])%dims[ndim-1]; /* Array, position, Circular wrapping */
				grid_values[ndim*5-1] = sub_matrix_sizes[ndim-1]*grid_values[ndim*4-1] + grid_values[ndim*5-2];
			}
		}
	}

	/*----------------*/
	/* Clean up       */
	/*----------------*/
	free(grid_values);
	free(kernel_values);
	free(matrix_centers);
	free(sub_matrix_sizes);
	if (weight != NULL && direction)
	{
		free(weighted_data_in);
	}

	return 0;
}


/* Calculated Modified Bessel Function Io(x)
   from Numerical Recipes in C */
double bessi0(double x)
{
	double ax,ans;
	double y;
	if ((ax=fabs(x)) < 3.75) {
		y=x/3.75;
		y*=y;
		ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
			    +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
	} else {
		y=3.75/ax;
      ans=(-0.02057706+y*(0.02635537+y*(-0.01647633+(y*0.00392377))));
      ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.01328592+y*(0.00225319+y*(-0.00157565+y*(0.00916281+y*ans)))));
	}
	return ans;
}

#undef SQR