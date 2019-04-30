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
  #define CASADI_PREFIX(ID) impl_ode_jac_x_xdot_u_ ## ID
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

static const casadi_int casadi_s0[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s2[23] = {4, 4, 0, 4, 8, 12, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

casadi_real casadi_sq(casadi_real x) { return x*x;}

/* casadi_impl_ode_jac_x_xdot_u:(i0[4],i1[4],i2)->(o0[4x4],o1[4x4],o2[4]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=0.;
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a0;
  if (res[0]!=0) res[0][2]=a0;
  if (res[0]!=0) res[0][3]=a0;
  if (res[0]!=0) res[0][4]=a0;
  if (res[0]!=0) res[0][5]=a0;
  a1=arg[0] ? arg[0][3] : 0;
  a2=casadi_sq(a1);
  a3=-8.0000000000000016e-02;
  a4=arg[0] ? arg[0][1] : 0;
  a5=cos(a4);
  a5=(a3*a5);
  a5=(a2*a5);
  a6=9.8100000000000009e-01;
  a7=cos(a4);
  a7=(a6*a7);
  a8=cos(a4);
  a8=(a7*a8);
  a9=sin(a4);
  a10=sin(a4);
  a10=(a6*a10);
  a10=(a9*a10);
  a8=(a8-a10);
  a5=(a5+a8);
  a8=1.1000000000000001e+00;
  a10=1.0000000000000001e-01;
  a11=cos(a4);
  a12=casadi_sq(a11);
  a12=(a10*a12);
  a12=(a8-a12);
  a5=(a5/a12);
  a13=sin(a4);
  a13=(a3*a13);
  a2=(a13*a2);
  a14=arg[2] ? arg[2][0] : 0;
  a2=(a2+a14);
  a7=(a7*a9);
  a2=(a2+a7);
  a2=(a2/a12);
  a2=(a2/a12);
  a11=(a11+a11);
  a7=sin(a4);
  a11=(a11*a7);
  a11=(a10*a11);
  a2=(a2*a11);
  a5=(a5-a2);
  a5=(-a5);
  if (res[0]!=0) res[0][6]=a5;
  a5=casadi_sq(a1);
  a2=cos(a4);
  a2=(a3*a2);
  a11=cos(a4);
  a11=(a2*a11);
  a7=sin(a4);
  a9=sin(a4);
  a3=(a3*a9);
  a3=(a7*a3);
  a11=(a11-a3);
  a11=(a5*a11);
  a3=sin(a4);
  a3=(a14*a3);
  a11=(a11-a3);
  a3=cos(a4);
  a3=(a6*a3);
  a11=(a11+a3);
  a3=9.8100000000000005e+00;
  a9=cos(a4);
  a9=(a3*a9);
  a11=(a11+a9);
  a9=8.0000000000000004e-01;
  a15=cos(a4);
  a16=casadi_sq(a15);
  a16=(a10*a16);
  a8=(a8-a16);
  a8=(a9*a8);
  a11=(a11/a8);
  a2=(a2*a7);
  a5=(a2*a5);
  a7=cos(a4);
  a14=(a14*a7);
  a5=(a5+a14);
  a14=sin(a4);
  a6=(a6*a14);
  a5=(a5+a6);
  a6=sin(a4);
  a3=(a3*a6);
  a5=(a5+a3);
  a5=(a5/a8);
  a5=(a5/a8);
  a15=(a15+a15);
  a4=sin(a4);
  a15=(a15*a4);
  a10=(a10*a15);
  a9=(a9*a10);
  a5=(a5*a9);
  a11=(a11-a5);
  a11=(-a11);
  if (res[0]!=0) res[0][7]=a11;
  a11=-1.;
  if (res[0]!=0) res[0][8]=a11;
  if (res[0]!=0) res[0][9]=a0;
  if (res[0]!=0) res[0][10]=a0;
  if (res[0]!=0) res[0][11]=a0;
  if (res[0]!=0) res[0][12]=a0;
  if (res[0]!=0) res[0][13]=a11;
  a11=(a1+a1);
  a13=(a13*a11);
  a13=(a13/a12);
  a13=(-a13);
  if (res[0]!=0) res[0][14]=a13;
  a1=(a1+a1);
  a2=(a2*a1);
  a2=(a2/a8);
  a2=(-a2);
  if (res[0]!=0) res[0][15]=a2;
  a2=1.;
  if (res[1]!=0) res[1][0]=a2;
  if (res[1]!=0) res[1][1]=a0;
  if (res[1]!=0) res[1][2]=a0;
  if (res[1]!=0) res[1][3]=a0;
  if (res[1]!=0) res[1][4]=a0;
  if (res[1]!=0) res[1][5]=a2;
  if (res[1]!=0) res[1][6]=a0;
  if (res[1]!=0) res[1][7]=a0;
  if (res[1]!=0) res[1][8]=a0;
  if (res[1]!=0) res[1][9]=a0;
  if (res[1]!=0) res[1][10]=a2;
  if (res[1]!=0) res[1][11]=a0;
  if (res[1]!=0) res[1][12]=a0;
  if (res[1]!=0) res[1][13]=a0;
  if (res[1]!=0) res[1][14]=a0;
  if (res[1]!=0) res[1][15]=a2;
  if (res[2]!=0) res[2][0]=a0;
  if (res[2]!=0) res[2][1]=a0;
  a12=(1./a12);
  a12=(-a12);
  if (res[2]!=0) res[2][2]=a12;
  a7=(a7/a8);
  a7=(-a7);
  if (res[2]!=0) res[2][3]=a7;
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_impl_ode_jac_x_xdot_u(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void casadi_impl_ode_jac_x_xdot_u_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_impl_ode_jac_x_xdot_u_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_impl_ode_jac_x_xdot_u_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int casadi_impl_ode_jac_x_xdot_u_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT const char* casadi_impl_ode_jac_x_xdot_u_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_impl_ode_jac_x_xdot_u_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_impl_ode_jac_x_xdot_u_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_impl_ode_jac_x_xdot_u_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s2;
    case 2: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_impl_ode_jac_x_xdot_u_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
