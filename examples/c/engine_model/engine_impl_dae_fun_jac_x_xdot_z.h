/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

int engine_impl_dae_fun_jac_x_xdot_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
void engine_impl_dae_fun_jac_x_xdot_z_incref(void);
void engine_impl_dae_fun_jac_x_xdot_z_decref(void);
casadi_int engine_impl_dae_fun_jac_x_xdot_z_n_in(void);
casadi_int engine_impl_dae_fun_jac_x_xdot_z_n_out(void);
const char* engine_impl_dae_fun_jac_x_xdot_z_name_in(casadi_int i);
const char* engine_impl_dae_fun_jac_x_xdot_z_name_out(casadi_int i);
const casadi_int* engine_impl_dae_fun_jac_x_xdot_z_sparsity_in(casadi_int i);
const casadi_int* engine_impl_dae_fun_jac_x_xdot_z_sparsity_out(casadi_int i);
int engine_impl_dae_fun_jac_x_xdot_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#ifdef __cplusplus
} /* extern "C" */
#endif
