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
  #define CASADI_PREFIX(ID) ls_costN_nm7_ ## ID
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

static const casadi_int casadi_s0[40] = {36, 1, 0, 36, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
static const casadi_int casadi_s1[75] = {36, 36, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};

/* ls_costN_nm7:(i0[36])->(o0[36],o1[36x36,36nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0;
  a0=arg[0] ? arg[0][0] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[0] ? arg[0][1] : 0;
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[0] ? arg[0][2] : 0;
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[0] ? arg[0][3] : 0;
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[0] ? arg[0][4] : 0;
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[0] ? arg[0][5] : 0;
  if (res[0]!=0) res[0][5]=a0;
  a0=arg[0] ? arg[0][6] : 0;
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[0] ? arg[0][7] : 0;
  if (res[0]!=0) res[0][7]=a0;
  a0=arg[0] ? arg[0][8] : 0;
  if (res[0]!=0) res[0][8]=a0;
  a0=arg[0] ? arg[0][9] : 0;
  if (res[0]!=0) res[0][9]=a0;
  a0=arg[0] ? arg[0][10] : 0;
  if (res[0]!=0) res[0][10]=a0;
  a0=arg[0] ? arg[0][11] : 0;
  if (res[0]!=0) res[0][11]=a0;
  a0=arg[0] ? arg[0][12] : 0;
  if (res[0]!=0) res[0][12]=a0;
  a0=arg[0] ? arg[0][13] : 0;
  if (res[0]!=0) res[0][13]=a0;
  a0=arg[0] ? arg[0][14] : 0;
  if (res[0]!=0) res[0][14]=a0;
  a0=arg[0] ? arg[0][15] : 0;
  if (res[0]!=0) res[0][15]=a0;
  a0=arg[0] ? arg[0][16] : 0;
  if (res[0]!=0) res[0][16]=a0;
  a0=arg[0] ? arg[0][17] : 0;
  if (res[0]!=0) res[0][17]=a0;
  a0=arg[0] ? arg[0][18] : 0;
  if (res[0]!=0) res[0][18]=a0;
  a0=arg[0] ? arg[0][19] : 0;
  if (res[0]!=0) res[0][19]=a0;
  a0=arg[0] ? arg[0][20] : 0;
  if (res[0]!=0) res[0][20]=a0;
  a0=arg[0] ? arg[0][21] : 0;
  if (res[0]!=0) res[0][21]=a0;
  a0=arg[0] ? arg[0][22] : 0;
  if (res[0]!=0) res[0][22]=a0;
  a0=arg[0] ? arg[0][23] : 0;
  if (res[0]!=0) res[0][23]=a0;
  a0=arg[0] ? arg[0][24] : 0;
  if (res[0]!=0) res[0][24]=a0;
  a0=arg[0] ? arg[0][25] : 0;
  if (res[0]!=0) res[0][25]=a0;
  a0=arg[0] ? arg[0][26] : 0;
  if (res[0]!=0) res[0][26]=a0;
  a0=arg[0] ? arg[0][27] : 0;
  if (res[0]!=0) res[0][27]=a0;
  a0=arg[0] ? arg[0][28] : 0;
  if (res[0]!=0) res[0][28]=a0;
  a0=arg[0] ? arg[0][29] : 0;
  if (res[0]!=0) res[0][29]=a0;
  a0=arg[0] ? arg[0][30] : 0;
  if (res[0]!=0) res[0][30]=a0;
  a0=arg[0] ? arg[0][31] : 0;
  if (res[0]!=0) res[0][31]=a0;
  a0=arg[0] ? arg[0][32] : 0;
  if (res[0]!=0) res[0][32]=a0;
  a0=arg[0] ? arg[0][33] : 0;
  if (res[0]!=0) res[0][33]=a0;
  a0=arg[0] ? arg[0][34] : 0;
  if (res[0]!=0) res[0][34]=a0;
  a0=arg[0] ? arg[0][35] : 0;
  if (res[0]!=0) res[0][35]=a0;
  a0=1.;
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
  if (res[1]!=0) res[1][12]=a0;
  if (res[1]!=0) res[1][13]=a0;
  if (res[1]!=0) res[1][14]=a0;
  if (res[1]!=0) res[1][15]=a0;
  if (res[1]!=0) res[1][16]=a0;
  if (res[1]!=0) res[1][17]=a0;
  if (res[1]!=0) res[1][18]=a0;
  if (res[1]!=0) res[1][19]=a0;
  if (res[1]!=0) res[1][20]=a0;
  if (res[1]!=0) res[1][21]=a0;
  if (res[1]!=0) res[1][22]=a0;
  if (res[1]!=0) res[1][23]=a0;
  if (res[1]!=0) res[1][24]=a0;
  if (res[1]!=0) res[1][25]=a0;
  if (res[1]!=0) res[1][26]=a0;
  if (res[1]!=0) res[1][27]=a0;
  if (res[1]!=0) res[1][28]=a0;
  if (res[1]!=0) res[1][29]=a0;
  if (res[1]!=0) res[1][30]=a0;
  if (res[1]!=0) res[1][31]=a0;
  if (res[1]!=0) res[1][32]=a0;
  if (res[1]!=0) res[1][33]=a0;
  if (res[1]!=0) res[1][34]=a0;
  if (res[1]!=0) res[1][35]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int ls_costN_nm7(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void ls_costN_nm7_incref(void) {
}

CASADI_SYMBOL_EXPORT void ls_costN_nm7_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int ls_costN_nm7_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int ls_costN_nm7_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT const char* ls_costN_nm7_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* ls_costN_nm7_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* ls_costN_nm7_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* ls_costN_nm7_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int ls_costN_nm7_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
