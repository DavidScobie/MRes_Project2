#ifndef GRID_H
#define GRID_H

#define GRID_DEFAULT_OVERSAMPLE_FACTOR 2.0
#define GRID_DEFAULT_KERNEL_WIDTH 4.0
#define GRID_DEFAULT_KERNEL_TABLE_STEPS 100000
#define GRID_DEFAULT_KERNEL_BETA 18.5547 
#define GRID_DEFAULT_KERNEL_SIGMA 0.3363
/*
#define GRID_KERNEL_GAUSSIAN
*/
/*
#define GRID_DEFAULT_KERNEL_BETA 13.9086
*/

/*
#define GRID_DEFAULT_KERNEL_BETA 18.5547 
*/
/* Struct for holding basic gridding parameters */
typedef struct {
    float over_sampling;
    float kernel_width;
    int kernel_table_steps;
    float kernel_beta;
    int kernel_samples;
    int transformed_kernel_samples;
    float* kernel_table;
    float* transformed_kernel_table;
} gridding_parameters;

typedef struct {
	int number_of_points;
	int number_of_dimensions;
	int* kernel_positions;
	int* grid_positions;
} point_vectors;

int genkt_grid( float *kt_pos, int npoints, float *data_in, float *weight, 
                float *data_out, int ndim, int *dims, 
                int *oversample_factors, float kernel_width, 
                float kernel_beta, int direction, int *fixed_dims);

int genkt_grid_2d(	gridding_parameters* parm, int *k_pos, int* grid_pos, int npoints, float *data_in, float *weight, 
					float *data_out, unsigned long *dims, int direction, int *fixed_dims);

int genkt_grid_3d(	gridding_parameters* parm, int *k_pos, int* grid_pos, int npoints, float *data_in, float *weight, 
					float *data_out, unsigned long *dims, int direction, int *fixed_dims);

void init_gridding_parameters(gridding_parameters* parm);

int calculate_kernel_tables(gridding_parameters* parm);

void delete_kernel_tables(gridding_parameters* parm);

int calculate_point_vectors(float* kt_pos, int npoints, int ndim, unsigned long* dims, gridding_parameters* parm, point_vectors* pv, int* fixed_dims);

void delete_point_vectors(point_vectors* pv);

double bessi0(double x);


#endif
