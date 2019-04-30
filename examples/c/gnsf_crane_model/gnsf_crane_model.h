#ifndef EXAMPLES_C_GNSF_CRANE_MODEL_GNSF_CRANE_MODEL_H_
#define EXAMPLES_C_GNSF_CRANE_MODEL_GNSF_CRANE_MODEL_H_

#include "acados/utils/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// used to import integers
int        casadi_get_ints_fun(const double** arg, double** res, int* iw, double* w, int mem);
int        casadi_get_ints_fun_work(int *, int *, int *, int *);
const int *casadi_get_ints_fun_sparsity_in(int);
const int *casadi_get_ints_fun_sparsity_out(int);
int        casadi_get_ints_fun_n_in();
int        casadi_get_ints_fun_n_out();

// used to import model matrices
int        casadi_get_matrices_fun(const double** arg, double** res, int* iw, double* w, int mem);
int        casadi_get_matrices_fun_work(int *, int *, int *, int *);
const int *casadi_get_matrices_fun_sparsity_in(int);
const int *casadi_get_matrices_fun_sparsity_out(int);
int        casadi_get_matrices_fun_n_in();
int        casadi_get_matrices_fun_n_out();

// phi_fun
int        casadi_phi_fun(const double** arg, double** res, int* iw, double* w, int mem);
int        casadi_phi_fun_work(int *, int *, int *, int *);
const int *casadi_phi_fun_sparsity_in(int);
const int *casadi_phi_fun_sparsity_out(int);
int        casadi_phi_fun_n_in();
int        casadi_phi_fun_n_out();

// phi_fun_jac_y
int        casadi_phi_fun_jac_y(const double** arg, double** res, int* iw, double* w, int mem);
int        casadi_phi_fun_jac_y_work(int *, int *, int *, int *);
const int *casadi_phi_fun_jac_y_sparsity_in(int);
const int *casadi_phi_fun_jac_y_sparsity_out(int);
int        casadi_phi_fun_jac_y_n_in();
int        casadi_phi_fun_jac_y_n_out();

// phi_jac_y_uhat
int        casadi_phi_jac_y_uhat(const double** arg, double** res, int* iw, double* w, int mem);
int        casadi_phi_jac_y_uhat_work(int *, int *, int *, int *);
const int *casadi_phi_jac_y_uhat_sparsity_in(int);
const int *casadi_phi_jac_y_uhat_sparsity_out(int);
int        casadi_phi_jac_y_uhat_n_in();
int        casadi_phi_jac_y_uhat_n_out();

// f_lo_fun_jac_x1k1uz
int        casadi_f_lo_fun_jac_x1k1uz(const double** arg, double** res, int* iw, double* w, int mem);
int        casadi_f_lo_fun_jac_x1k1uz_work(int *, int *, int *, int *);
const int *casadi_f_lo_fun_jac_x1k1uz_sparsity_in(int);
const int *casadi_f_lo_fun_jac_x1k1uz_sparsity_out(int);
int        casadi_f_lo_fun_jac_x1k1uz_n_in();
int        casadi_f_lo_fun_jac_x1k1uz_n_out();

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // EXAMPLES_C_CRANE_MODEL_CRANE_MODEL_H_