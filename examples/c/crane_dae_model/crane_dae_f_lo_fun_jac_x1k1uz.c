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
  #define CASADI_PREFIX(ID) crane_dae_f_lo_fun_jac_x1k1uz_ ## ID
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
#define casadi_sq CASADI_PREFIX(sq)

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

static const casadi_int casadi_s0[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s3[27] = {6, 14, 0, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 9, 10, 10, 10, 3, 4, 5, 4, 5, 5, 2, 3, 5, 4};

casadi_real casadi_sq(casadi_real x) { return x*x;}

/* crane_dae_f_lo_fun_jac_x1k1uz:(i0[5],i1[5],i2[2],i3[2])->(o0[6],o1[6x14,10nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7, a8;
  a0=0.;
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[3] ? arg[3][0] : 0;
  if (res[0]!=0) res[0][2]=a0;
  a1=casadi_sq(a0);
  a2=arg[0] ? arg[0][0] : 0;
  a3=casadi_sq(a2);
  a1=(a1+a3);
  a3=cos(a2);
  a1=(a1+a3);
  if (res[0]!=0) res[0][3]=a1;
  a1=arg[0] ? arg[0][3] : 0;
  a3=casadi_sq(a1);
  a4=8.;
  a3=(a3/a4);
  a4=arg[3] ? arg[3][1] : 0;
  a5=sin(a4);
  a3=(a3+a5);
  a3=(a2+a3);
  a3=(-a3);
  if (res[0]!=0) res[0][4]=a3;
  a3=arg[0] ? arg[0][4] : 0;
  a5=1.0000000000000001e-01;
  a3=(a3+a5);
  a5=cos(a3);
  a6=arg[1] ? arg[1][0] : 0;
  a7=arg[0] ? arg[0][1] : 0;
  a8=(a0*a7);
  a6=(a6-a8);
  a8=casadi_sq(a6);
  a5=(a5+a8);
  a5=(-a5);
  if (res[0]!=0) res[0][5]=a5;
  a5=(a2+a2);
  a2=sin(a2);
  a5=(a5-a2);
  if (res[1]!=0) res[1][0]=a5;
  a5=-1.;
  if (res[1]!=0) res[1][1]=a5;
  a5=(a6+a6);
  a5=(a5*a0);
  if (res[1]!=0) res[1][2]=a5;
  a5=1.2500000000000000e-01;
  a1=(a1+a1);
  a5=(a5*a1);
  a5=(-a5);
  if (res[1]!=0) res[1][3]=a5;
  a3=sin(a3);
  if (res[1]!=0) res[1][4]=a3;
  a3=(a6+a6);
  a3=(-a3);
  if (res[1]!=0) res[1][5]=a3;
  a3=1.;
  if (res[1]!=0) res[1][6]=a3;
  a0=(a0+a0);
  if (res[1]!=0) res[1][7]=a0;
  a6=(a6+a6);
  a6=(a6*a7);
  if (res[1]!=0) res[1][8]=a6;
  a4=cos(a4);
  a4=(-a4);
  if (res[1]!=0) res[1][9]=a4;
  return 0;
}

CASADI_SYMBOL_EXPORT int crane_dae_f_lo_fun_jac_x1k1uz(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void crane_dae_f_lo_fun_jac_x1k1uz_incref(void) {
}

CASADI_SYMBOL_EXPORT void crane_dae_f_lo_fun_jac_x1k1uz_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int crane_dae_f_lo_fun_jac_x1k1uz_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int crane_dae_f_lo_fun_jac_x1k1uz_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT const char* crane_dae_f_lo_fun_jac_x1k1uz_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* crane_dae_f_lo_fun_jac_x1k1uz_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* crane_dae_f_lo_fun_jac_x1k1uz_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* crane_dae_f_lo_fun_jac_x1k1uz_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int crane_dae_f_lo_fun_jac_x1k1uz_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
