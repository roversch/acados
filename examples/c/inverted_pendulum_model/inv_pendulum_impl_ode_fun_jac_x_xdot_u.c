/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CODEGEN_PREFIX
  #define NAMESPACE_CONCAT(NS, ID) _NAMESPACE_CONCAT(NS, ID)
  #define _NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) inv_pendulum_impl_ode_fun_jac_x_xdot_u_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s2[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s3[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s4[20] = {11, 6, 0, 2, 4, 6, 8, 11, 11, 6, 7, 5, 7, 0, 6, 1, 5, 5, 6, 10};
static const casadi_int casadi_s5[15] = {11, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 10};
static const casadi_int casadi_s6[6] = {11, 1, 0, 2, 7, 8};

/* inv_pendulum_impl_ode_fun_jac_x_xdot_u:(i0[6],i1[6],i2,i3[5])->(o0[11],o1[11x6,11nz],o2[11x6,6nz],o3[11x1,2nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[1] ? arg[1][0] : 0;
  a1=arg[0] ? arg[0][2] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[1] ? arg[1][1] : 0;
  a2=arg[0] ? arg[0][3] : 0;
  a0=(a0-a2);
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[1] ? arg[1][2] : 0;
  a3=arg[3] ? arg[3][0] : 0;
  a0=(a0-a3);
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[1] ? arg[1][3] : 0;
  a4=arg[3] ? arg[3][1] : 0;
  a0=(a0-a4);
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[1] ? arg[1][4] : 0;
  a5=arg[3] ? arg[3][2] : 0;
  a0=(a0-a5);
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[0] ? arg[0][4] : 0;
  a6=(a2*a0);
  a6=(a3+a6);
  a7=arg[0] ? arg[0][1] : 0;
  a8=(a7*a5);
  a6=(a6+a8);
  if (res[0]!=0) res[0][5]=a6;
  a6=(a1*a0);
  a6=(a4-a6);
  a8=arg[0] ? arg[0][0] : 0;
  a9=(a8*a5);
  a6=(a6-a9);
  if (res[0]!=0) res[0][6]=a6;
  a6=1.0000000000000001e-01;
  a6=(a6*a5);
  a9=3.5000000000000000e+00;
  a6=(a6-a9);
  a9=arg[3] ? arg[3][3] : 0;
  a10=arg[2] ? arg[2][0] : 0;
  a11=(a9+a10);
  a12=(a11*a7);
  a6=(a6-a12);
  a12=arg[3] ? arg[3][4] : 0;
  a8=(a12*a8);
  a6=(a6+a8);
  if (res[0]!=0) res[0][7]=a6;
  a6=2.;
  a3=(a6*a3);
  a9=(a9+a10);
  a3=(a3-a9);
  if (res[0]!=0) res[0][8]=a3;
  a6=(a6*a4);
  a4=1.9620000000000001e+01;
  a6=(a6+a4);
  a6=(a6-a12);
  if (res[0]!=0) res[0][9]=a6;
  a6=arg[1] ? arg[1][5] : 0;
  a6=(a6-a0);
  if (res[0]!=0) res[0][10]=a6;
  a6=(-a5);
  if (res[1]!=0) res[1][0]=a6;
  if (res[1]!=0) res[1][1]=a12;
  if (res[1]!=0) res[1][2]=a5;
  a11=(-a11);
  if (res[1]!=0) res[1][3]=a11;
  a11=-1.;
  if (res[1]!=0) res[1][4]=a11;
  a5=(-a0);
  if (res[1]!=0) res[1][5]=a5;
  if (res[1]!=0) res[1][6]=a11;
  if (res[1]!=0) res[1][7]=a0;
  if (res[1]!=0) res[1][8]=a2;
  a1=(-a1);
  if (res[1]!=0) res[1][9]=a1;
  if (res[1]!=0) res[1][10]=a11;
  a1=1.;
  if (res[2]!=0) res[2][0]=a1;
  if (res[2]!=0) res[2][1]=a1;
  if (res[2]!=0) res[2][2]=a1;
  if (res[2]!=0) res[2][3]=a1;
  if (res[2]!=0) res[2][4]=a1;
  if (res[2]!=0) res[2][5]=a1;
  a7=(-a7);
  if (res[3]!=0) res[3][0]=a7;
  if (res[3]!=0) res[3][1]=a11;
  return 0;
}

CASADI_SYMBOL_EXPORT int inv_pendulum_impl_ode_fun_jac_x_xdot_u(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void inv_pendulum_impl_ode_fun_jac_x_xdot_u_incref(void) {
}

CASADI_SYMBOL_EXPORT void inv_pendulum_impl_ode_fun_jac_x_xdot_u_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int inv_pendulum_impl_ode_fun_jac_x_xdot_u_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int inv_pendulum_impl_ode_fun_jac_x_xdot_u_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT const char* inv_pendulum_impl_ode_fun_jac_x_xdot_u_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* inv_pendulum_impl_ode_fun_jac_x_xdot_u_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* inv_pendulum_impl_ode_fun_jac_x_xdot_u_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* inv_pendulum_impl_ode_fun_jac_x_xdot_u_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int inv_pendulum_impl_ode_fun_jac_x_xdot_u_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
