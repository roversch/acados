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
  #define CASADI_PREFIX(ID) get_matrices_fun_ ## ID
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

static const casadi_int casadi_s0[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s1[15] = {3, 3, 0, 3, 6, 9, 0, 1, 2, 0, 1, 2, 0, 1, 2};
static const casadi_int casadi_s2[19] = {3, 4, 0, 3, 6, 9, 12, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
static const casadi_int casadi_s3[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s4[3] = {3, 0, 0};
static const casadi_int casadi_s5[23] = {4, 4, 0, 4, 8, 12, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s6[3] = {0, 0, 0};

/* casadi_get_matrices_fun:(i0)->(o0[3x3],o1[3x4],o2[3],o3[3x3],o4[3x3],o5[3x3],o6[3x0],o7[4x4],o8[],o9[3],o10[]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1;
  a0=0.;
  if (res[0]!=0) res[0][0]=a0;
  a1=1.;
  if (res[0]!=0) res[0][1]=a1;
  if (res[0]!=0) res[0][2]=a0;
  if (res[0]!=0) res[0][3]=a0;
  if (res[0]!=0) res[0][4]=a0;
  if (res[0]!=0) res[0][5]=a0;
  if (res[0]!=0) res[0][6]=a0;
  if (res[0]!=0) res[0][7]=a0;
  if (res[0]!=0) res[0][8]=a0;
  if (res[1]!=0) res[1][0]=a0;
  if (res[1]!=0) res[1][1]=a0;
  if (res[1]!=0) res[1][2]=a0;
  if (res[1]!=0) res[1][3]=a0;
  if (res[1]!=0) res[1][4]=a0;
  if (res[1]!=0) res[1][5]=a0;
  if (res[1]!=0) res[1][6]=a0;
  if (res[1]!=0) res[1][7]=a0;
  if (res[1]!=0) res[1][8]=a0;
  if (res[1]!=0) res[1][9]=a0;
  if (res[1]!=0) res[1][10]=a0;
  if (res[1]!=0) res[1][11]=a0;
  if (res[2]!=0) res[2][0]=a1;
  if (res[2]!=0) res[2][1]=a0;
  if (res[2]!=0) res[2][2]=a0;
  if (res[3]!=0) res[3][0]=a1;
  if (res[3]!=0) res[3][1]=a0;
  if (res[3]!=0) res[3][2]=a0;
  if (res[3]!=0) res[3][3]=a0;
  if (res[3]!=0) res[3][4]=a1;
  if (res[3]!=0) res[3][5]=a0;
  if (res[3]!=0) res[3][6]=a0;
  if (res[3]!=0) res[3][7]=a0;
  if (res[3]!=0) res[3][8]=a1;
  if (res[4]!=0) res[4][0]=a1;
  if (res[4]!=0) res[4][1]=a0;
  if (res[4]!=0) res[4][2]=a0;
  if (res[4]!=0) res[4][3]=a0;
  if (res[4]!=0) res[4][4]=a1;
  if (res[4]!=0) res[4][5]=a0;
  if (res[4]!=0) res[4][6]=a0;
  if (res[4]!=0) res[4][7]=a0;
  if (res[4]!=0) res[4][8]=a1;
  if (res[5]!=0) res[5][0]=a0;
  if (res[5]!=0) res[5][1]=a0;
  if (res[5]!=0) res[5][2]=a0;
  if (res[5]!=0) res[5][3]=a0;
  if (res[5]!=0) res[5][4]=a0;
  if (res[5]!=0) res[5][5]=a0;
  if (res[5]!=0) res[5][6]=a0;
  if (res[5]!=0) res[5][7]=a0;
  if (res[5]!=0) res[5][8]=a0;
  if (res[7]!=0) res[7][0]=a1;
  if (res[7]!=0) res[7][1]=a0;
  if (res[7]!=0) res[7][2]=a0;
  if (res[7]!=0) res[7][3]=a0;
  if (res[7]!=0) res[7][4]=a0;
  if (res[7]!=0) res[7][5]=a1;
  if (res[7]!=0) res[7][6]=a0;
  if (res[7]!=0) res[7][7]=a0;
  if (res[7]!=0) res[7][8]=a0;
  if (res[7]!=0) res[7][9]=a0;
  if (res[7]!=0) res[7][10]=a1;
  if (res[7]!=0) res[7][11]=a0;
  if (res[7]!=0) res[7][12]=a0;
  if (res[7]!=0) res[7][13]=a0;
  if (res[7]!=0) res[7][14]=a0;
  if (res[7]!=0) res[7][15]=a1;
  if (res[9]!=0) res[9][0]=a0;
  if (res[9]!=0) res[9][1]=a0;
  if (res[9]!=0) res[9][2]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_get_matrices_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void casadi_get_matrices_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_get_matrices_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_get_matrices_fun_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int casadi_get_matrices_fun_n_out(void) { return 11;}

CASADI_SYMBOL_EXPORT const char* casadi_get_matrices_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_get_matrices_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    case 5: return "o5";
    case 6: return "o6";
    case 7: return "o7";
    case 8: return "o8";
    case 9: return "o9";
    case 10: return "o10";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_get_matrices_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_get_matrices_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    case 1: return casadi_s2;
    case 2: return casadi_s3;
    case 3: return casadi_s1;
    case 4: return casadi_s1;
    case 5: return casadi_s1;
    case 6: return casadi_s4;
    case 7: return casadi_s5;
    case 8: return casadi_s6;
    case 9: return casadi_s3;
    case 10: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_get_matrices_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 11;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
