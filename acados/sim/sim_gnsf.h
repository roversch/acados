/*
 *    This file is part of acados.
 *
 *    acados is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    acados is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with acados; if not, write to the Free Software Foundation,
 *    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *    Author: Jonathan Frey
 */

#ifndef ACADOS_SIM_SIM_GNSF_H_
#define ACADOS_SIM_SIM_GNSF_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include "acados/utils/timing.h"
#include "acados/utils/types.h"

#include "acados/sim/sim_common.h"

#include "blasfeo/blasfeo_common.h"
#include "blasfeo/blasfeo_d_aux.h"
#include "blasfeo/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/blasfeo_d_blas.h"
#include "blasfeo/blasfeo_d_kernel.h"
#include "blasfeo/blasfeo_i_aux_ext_dep.h"
#include "blasfeo/blasfeo_target.h"

typedef struct
{
    int nx;
    int nu;
    int nz;
    int nx1;
    int nz1;
    int n_out;
    int ny;
    int nuhat;

} sim_gnsf_dims;

typedef struct
{
    /* external functions */
    // phi: nonlinearity function
    external_function_generic *phi_fun;
    external_function_generic *phi_fun_jac_y;
    external_function_generic *phi_jac_y_uhat;
    // f_lo: linear output function
    external_function_generic *f_lo_fun_jac_x1_x1dot_u_z;

    /* model defining matrices */
    double *A;
    double *B;
    double *C;
    double *E;

    double *L_x;
    double *L_xdot;
    double *L_z;
    double *L_u;

    double *A_LO;
    double *E_LO;

    /* constant vector */
    double *c;

} gnsf_model;

// pre_workspace - workspace used in the precomputation phase
typedef struct
{
    struct blasfeo_dmat E11;
    struct blasfeo_dmat E12;
    struct blasfeo_dmat E21;
    struct blasfeo_dmat E22;

    struct blasfeo_dmat A1;
    struct blasfeo_dmat A2;
    struct blasfeo_dmat B1;
    struct blasfeo_dmat B2;
    struct blasfeo_dmat C1;
    struct blasfeo_dmat C2;

    struct blasfeo_dmat AA1;
    struct blasfeo_dmat AA2;
    struct blasfeo_dmat BB1;
    struct blasfeo_dmat BB2;
    struct blasfeo_dmat CC1;
    struct blasfeo_dmat CC2;
    struct blasfeo_dmat DD1;
    struct blasfeo_dmat DD2;
    struct blasfeo_dmat EE1;
    struct blasfeo_dmat EE2;

    struct blasfeo_dmat QQ1;

    struct blasfeo_dmat LLZ;
    struct blasfeo_dmat LLx;
    struct blasfeo_dmat LLK;

    int *ipivEE1;  // index of pivot vector
    int *ipivEE2;
    int *ipivQQ1;

    // for algebraic sensitivity propagation
    struct blasfeo_dmat Q1;

    // for constant term in NSF
    struct blasfeo_dvec cc1;
    struct blasfeo_dvec cc2;

} gnsf_pre_workspace;

// workspace
typedef struct
{
    double *Z_work;  // used to perform computations to get out->zn

    int *ipiv;  // index of pivot vector

    struct blasfeo_dvec *vv_traj;
    struct blasfeo_dvec *yy_traj;
    struct blasfeo_dmat *f_LO_jac_traj;

    struct blasfeo_dvec K2_val;
    struct blasfeo_dvec x0_traj;
    struct blasfeo_dvec res_val;
    struct blasfeo_dvec u0;
    struct blasfeo_dvec lambda;
    struct blasfeo_dvec lambda_old;

    struct blasfeo_dvec yyu;
    struct blasfeo_dvec yyss;
    struct blasfeo_dvec y_one_stage;

    struct blasfeo_dvec K1_val;
    struct blasfeo_dvec f_LO_val;
    struct blasfeo_dvec x1_stage_val;
    struct blasfeo_dvec Z1_val;

    struct blasfeo_dvec K1u;
    struct blasfeo_dvec Zu;
    struct blasfeo_dvec ALOtimesx02;

    struct blasfeo_dvec uhat;

    struct blasfeo_dmat J_r_vv;
    struct blasfeo_dmat J_r_x1u;

    struct blasfeo_dmat dK1_dx1;
    struct blasfeo_dmat dK1_du;
    struct blasfeo_dmat dZ_dx1;
    struct blasfeo_dmat dZ_du;
    struct blasfeo_dmat J_G2_K1;

    struct blasfeo_dmat dK2_dx1;
    struct blasfeo_dmat dK2_du;
    struct blasfeo_dmat dK2_dvv;
    struct blasfeo_dmat dxf_dwn;
    struct blasfeo_dmat S_forw_new;
    struct blasfeo_dmat S_forw;

    struct blasfeo_dmat dPsi_dvv;
    struct blasfeo_dmat dPsi_dx;
    struct blasfeo_dmat dPsi_du;

    struct blasfeo_dmat dPHI_dyuhat;

    // memory only available if (opts->sens_algebraic)
    struct blasfeo_dvec x0dot_1;
    struct blasfeo_dvec z0_1;
    struct blasfeo_dmat dz10_dx1u;  // (nz1) * (nx1+nu);
    struct blasfeo_dmat dr0_dvv0;  // (n_out * n_out)
    struct blasfeo_dmat f_LO_jac0; // (nx2+nz2) * (2*nx1 + nz1 + nu)
    struct blasfeo_dmat sens_z2_rhs; // (nx2 + nz2) * (nx1 + nu)
    int *ipiv_vv0;

} gnsf_workspace;

// memory
typedef struct
{
    // simulation time for one step
    double dt;

    // (scaled) butcher table
    double *A_dt;
    double *b_dt;
    double *c_butcher;

    // value used to initialize integration variables - corresponding to value of phi
    double *phi_guess;  //  n_out

    // precomputed matrices
    struct blasfeo_dmat KKv;
    struct blasfeo_dmat KKx;
    struct blasfeo_dmat KKu;

    struct blasfeo_dmat YYv;
    struct blasfeo_dmat YYx;
    struct blasfeo_dmat YYu;

    struct blasfeo_dmat ZZv;
    struct blasfeo_dmat ZZx;
    struct blasfeo_dmat ZZu;

    struct blasfeo_dmat ALO;
    struct blasfeo_dmat M2_LU;
    int *ipivM2;

    struct blasfeo_dmat dK2_dx2;

    struct blasfeo_dmat Lu;

    // precomputed vectors for constant term in NSF
    struct blasfeo_dvec KK0;
    struct blasfeo_dvec YY0;
    struct blasfeo_dvec ZZ0;

    // for algebraic sensitivities only;
    struct blasfeo_dmat *Z0x;
    struct blasfeo_dmat *Z0u;
    struct blasfeo_dmat *Z0v;

    struct blasfeo_dmat *Y0x;
    struct blasfeo_dmat *Y0u;
    struct blasfeo_dmat *Y0v;

    struct blasfeo_dmat *K0x;
    struct blasfeo_dmat *K0u;
    struct blasfeo_dmat *K0v;

    struct blasfeo_dmat *ELO_LU;
    int *ipiv_ELO;
    struct blasfeo_dmat *ELO_inv_ALO;

    struct blasfeo_dmat *Lx;
    struct blasfeo_dmat *Lxdot;
    struct blasfeo_dmat *Lz;

} sim_gnsf_memory;

// gnsf dims
int sim_gnsf_dims_calculate_size();
void *sim_gnsf_dims_assign(void *config_, void *raw_memory);

// get & set functions
void sim_gnsf_set_nx(void *dims_, int nx);
void sim_gnsf_set_nu(void *dims_, int nu);
void sim_gnsf_set_nz(void *dims_, int nz);

void sim_gnsf_get_nx(void *dims_, int *nx);
void sim_gnsf_get_nu(void *dims_, int *nu);
void sim_gnsf_get_nz(void *dims_, int *nz);

// opts
int sim_gnsf_opts_calculate_size(void *config, void *dims);
void *sim_gnsf_opts_assign(void *config, void *dims, void *raw_memory);
void sim_gnsf_opts_initialize_default(void *config, void *dims, void *opts_);
void sim_gnsf_opts_update(void *config_, void *dims, void *opts_);

// model
int sim_gnsf_model_calculate_size(void *config, void *dims_);
void *sim_gnsf_model_assign(void *config, void *dims_, void *raw_memory);
int sim_gnsf_model_set_function(void *model_, sim_function_t fun_type, void *fun);

// import
void sim_gnsf_import_matrices(sim_gnsf_dims *dims, gnsf_model *model,
                              external_function_generic *get_matrices_fun);

// precomputation
void sim_gnsf_precompute(void *config, sim_gnsf_dims *dims, gnsf_model *model, sim_rk_opts *opts,
                         void *mem_, void *work_, double T);

// workspace & memory
int sim_gnsf_workspace_calculate_size(void *config, void *dims_, void *args);
int sim_gnsf_memory_calculate_size(void *config, void *dims_, void *opts_);
void *sim_gnsf_memory_assign(void *config, void *dims_, void *opts_, void *raw_memory);

// interface
void sim_gnsf_config_initialize_default(void *config_);

// integrator
int sim_gnsf(void *config, sim_in *in, sim_out *out, void *opts, void *mem_, void *work_);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SIM_SIM_GNSF_H_
