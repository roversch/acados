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
  #define CASADI_PREFIX(ID) pendulum_ode_expl_ode_fun_ ## ID
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

/* pendulum_ode_expl_ode_fun:(i0[4],i1)->(o0[4]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7;
  a0=arg[0] ? arg[0][1] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a0=-8.0000000000000016e-02;
  a1=arg[0] ? arg[0][2] : 0;
  a2=sin(a1);
  a2=(a0*a2);
  a3=arg[0] ? arg[0][3] : 0;
  a2=(a2*a3);
  a2=(a2*a3);
  a4=9.8100000000000009e-01;
  a5=cos(a1);
  a4=(a4*a5);
  a5=sin(a1);
  a4=(a4*a5);
  a2=(a2+a4);
  a4=arg[1] ? arg[1][0] : 0;
  a2=(a2+a4);
  a5=1.1000000000000001e+00;
  a6=1.0000000000000001e-01;
  a7=cos(a1);
  a6=(a6*a7);
  a7=cos(a1);
  a6=(a6*a7);
  a5=(a5-a6);
  a2=(a2/a5);
  if (res[0]!=0) res[0][1]=a2;
  if (res[0]!=0) res[0][2]=a3;
  a2=cos(a1);
  a0=(a0*a2);
  a2=sin(a1);
  a0=(a0*a2);
  a0=(a0*a3);
  a0=(a0*a3);
  a3=cos(a1);
  a4=(a4*a3);
  a0=(a0+a4);
  a4=1.0791000000000002e+01;
  a1=sin(a1);
  a4=(a4*a1);
  a0=(a0+a4);
  a4=8.0000000000000004e-01;
  a4=(a4*a5);
  a0=(a0/a4);
  if (res[0]!=0) res[0][3]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void pendulum_ode_expl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void pendulum_ode_expl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int pendulum_ode_expl_ode_fun_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int pendulum_ode_expl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT const char* pendulum_ode_expl_ode_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* pendulum_ode_expl_ode_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* pendulum_ode_expl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* pendulum_ode_expl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
