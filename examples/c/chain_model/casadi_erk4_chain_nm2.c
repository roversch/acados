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
  #define CASADI_PREFIX(ID) casadi_erk4_chain_nm2_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_project CASADI_PREFIX(project)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s11 CASADI_PREFIX(s11)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)
#define casadi_trans CASADI_PREFIX(trans)

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
static const casadi_int casadi_s1[7] = {6, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s2[24] = {6, 3, 0, 6, 12, 18, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s3[12] = {6, 3, 0, 3, 6, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s4[15] = {3, 3, 0, 3, 6, 9, 0, 1, 2, 0, 1, 2, 0, 1, 2};
static const casadi_int casadi_s5[9] = {3, 3, 0, 0, 0, 3, 0, 1, 2};
static const casadi_int casadi_s6[7] = {6, 1, 0, 3, 3, 4, 5};
static const casadi_int casadi_s7[6] = {1, 4, 7, 9, 11, 13};
static const casadi_int casadi_s8[6] = {2, 5, 8, 10, 12, 14};
static const casadi_int casadi_s9[27] = {6, 9, 0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 0, 1, 2, 0, 3, 1, 4, 2, 5, 0, 3, 1, 4, 2, 5};
static const casadi_int casadi_s10[24] = {9, 6, 0, 3, 6, 9, 11, 13, 15, 0, 3, 6, 1, 4, 7, 2, 5, 8, 3, 6, 4, 7, 5, 8};
static const casadi_int casadi_s11[7] = {3, 1, 0, 3, 0, 1, 2};

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_project(const casadi_real* x, const casadi_int* sp_x, casadi_real* y, const casadi_int* sp_y, casadi_real* w) {
  casadi_int ncol_x, ncol_y, i, el;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y;
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  for (i=0; i<ncol_x; ++i) {
    for (el=colind_y[i]; el<colind_y[i+1]; ++el) w[row_y[el]] = 0;
    for (el=colind_x[i]; el<colind_x[i+1]; ++el) w[row_x[el]] = x[el];
    for (el=colind_y[i]; el<colind_y[i+1]; ++el) y[el] = w[row_y[el]];
  }
}

void casadi_trans(const casadi_real* x, const casadi_int* sp_x, casadi_real* y,
    const casadi_int* sp_y, casadi_int* tmp) {
  casadi_int ncol_x, nnz_x, ncol_y, k;
  const casadi_int* row_x, *colind_y;
  ncol_x = sp_x[1];
  nnz_x = sp_x[2 + ncol_x];
  row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}

/* f:(i0[6],i1[3])->(o0[6]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0;
  a0=arg[0] ? arg[0][3] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[0] ? arg[0][4] : 0;
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[0] ? arg[0][5] : 0;
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[1] ? arg[1][0] : 0;
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[1] ? arg[1][1] : 0;
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[1] ? arg[1][2] : 0;
  if (res[0]!=0) res[0][5]=a0;
  return 0;
}

/* fwd3_f:(i0[6],i1[3],out_o0[6x1,0nz],fwd_i0[6x3],fwd_i1[3x3])->(fwd_o0[6x3]) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0;
  a0=arg[3] ? arg[3][3] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[3] ? arg[3][4] : 0;
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[3] ? arg[3][5] : 0;
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[4] ? arg[4][0] : 0;
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[4] ? arg[4][1] : 0;
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[4] ? arg[4][2] : 0;
  if (res[0]!=0) res[0][5]=a0;
  a0=arg[3] ? arg[3][9] : 0;
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[3] ? arg[3][10] : 0;
  if (res[0]!=0) res[0][7]=a0;
  a0=arg[3] ? arg[3][11] : 0;
  if (res[0]!=0) res[0][8]=a0;
  a0=arg[4] ? arg[4][3] : 0;
  if (res[0]!=0) res[0][9]=a0;
  a0=arg[4] ? arg[4][4] : 0;
  if (res[0]!=0) res[0][10]=a0;
  a0=arg[4] ? arg[4][5] : 0;
  if (res[0]!=0) res[0][11]=a0;
  a0=arg[3] ? arg[3][15] : 0;
  if (res[0]!=0) res[0][12]=a0;
  a0=arg[3] ? arg[3][16] : 0;
  if (res[0]!=0) res[0][13]=a0;
  a0=arg[3] ? arg[3][17] : 0;
  if (res[0]!=0) res[0][14]=a0;
  a0=arg[4] ? arg[4][6] : 0;
  if (res[0]!=0) res[0][15]=a0;
  a0=arg[4] ? arg[4][7] : 0;
  if (res[0]!=0) res[0][16]=a0;
  a0=arg[4] ? arg[4][8] : 0;
  if (res[0]!=0) res[0][17]=a0;
  return 0;
}

/* casadi_erk4_chain_nm2:(x0[6],p[3])->(xf[6],sensxu[6x9,15nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real **res1=res+2, *rr, *ss;
  const casadi_int *cii;
  const casadi_real **arg1=arg+2, *cr, *cs;
  casadi_real *w0=w+6, w1, *w2=w+13, *w3=w+16, w4, *w5=w+23, *w6=w+29, *w7=w+35, w8, *w9=w+42, w10, w11, w12, *w13=w+51, *w14=w+57, *w15=w+63, w16, *w17=w+70, w18, *w19=w+77, *w20=w+92, *w21=w+95, *w23=w+98, *w25=w+101, *w27=w+119, *w28=w+122, *w29=w+131, *w30=w+149, *w31=w+155, *w32=w+161, *w33=w+167, *w34=w+173, *w35=w+179, *w36=w+185;
  /* #0: @0 = input[0][0] */
  casadi_copy(arg[0], 6, w0);
  /* #1: @1 = 0.0208333 */
  w1 = 2.0833333333333332e-02;
  /* #2: @2 = input[1][0] */
  casadi_copy(arg[1], 3, w2);
  /* #3: @3 = f(@0, @2) */
  arg1[0]=w0;
  arg1[1]=w2;
  res1[0]=w3;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #4: @4 = 0.0625 */
  w4 = 6.2500000000000000e-02;
  /* #5: @5 = (@4*@3) */
  for (i=0, rr=w5, cs=w3; i<6; ++i) (*rr++)  = (w4*(*cs++));
  /* #6: @5 = (@0+@5) */
  for (i=0, rr=w5, cr=w0, cs=w5; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #7: @6 = f(@5, @2) */
  arg1[0]=w5;
  arg1[1]=w2;
  res1[0]=w6;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #8: @7 = (2.*@6) */
  for (i=0, rr=w7, cs=w6; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #9: @3 = (@3+@7) */
  for (i=0, rr=w3, cs=w7; i<6; ++i) (*rr++) += (*cs++);
  /* #10: @8 = 0.0625 */
  w8 = 6.2500000000000000e-02;
  /* #11: @6 = (@8*@6) */
  for (i=0, rr=w6, cs=w6; i<6; ++i) (*rr++)  = (w8*(*cs++));
  /* #12: @6 = (@0+@6) */
  for (i=0, rr=w6, cr=w0, cs=w6; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #13: @7 = f(@6, @2) */
  arg1[0]=w6;
  arg1[1]=w2;
  res1[0]=w7;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #14: @9 = (2.*@7) */
  for (i=0, rr=w9, cs=w7; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #15: @3 = (@3+@9) */
  for (i=0, rr=w3, cs=w9; i<6; ++i) (*rr++) += (*cs++);
  /* #16: @10 = 0.125 */
  w10 = 1.2500000000000000e-01;
  /* #17: @7 = (@10*@7) */
  for (i=0, rr=w7, cs=w7; i<6; ++i) (*rr++)  = (w10*(*cs++));
  /* #18: @7 = (@0+@7) */
  for (i=0, rr=w7, cr=w0, cs=w7; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #19: @9 = f(@7, @2) */
  arg1[0]=w7;
  arg1[1]=w2;
  res1[0]=w9;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #20: @3 = (@3+@9) */
  for (i=0, rr=w3, cs=w9; i<6; ++i) (*rr++) += (*cs++);
  /* #21: @3 = (@1*@3) */
  for (i=0, rr=w3, cs=w3; i<6; ++i) (*rr++)  = (w1*(*cs++));
  /* #22: @3 = (@0+@3) */
  for (i=0, rr=w3, cr=w0, cs=w3; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #23: @11 = 0.0208333 */
  w11 = 2.0833333333333332e-02;
  /* #24: @9 = f(@3, @2) */
  arg1[0]=w3;
  arg1[1]=w2;
  res1[0]=w9;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #25: @12 = 0.0625 */
  w12 = 6.2500000000000000e-02;
  /* #26: @13 = (@12*@9) */
  for (i=0, rr=w13, cs=w9; i<6; ++i) (*rr++)  = (w12*(*cs++));
  /* #27: @13 = (@3+@13) */
  for (i=0, rr=w13, cr=w3, cs=w13; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #28: @14 = f(@13, @2) */
  arg1[0]=w13;
  arg1[1]=w2;
  res1[0]=w14;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #29: @15 = (2.*@14) */
  for (i=0, rr=w15, cs=w14; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #30: @9 = (@9+@15) */
  for (i=0, rr=w9, cs=w15; i<6; ++i) (*rr++) += (*cs++);
  /* #31: @16 = 0.0625 */
  w16 = 6.2500000000000000e-02;
  /* #32: @14 = (@16*@14) */
  for (i=0, rr=w14, cs=w14; i<6; ++i) (*rr++)  = (w16*(*cs++));
  /* #33: @14 = (@3+@14) */
  for (i=0, rr=w14, cr=w3, cs=w14; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #34: @15 = f(@14, @2) */
  arg1[0]=w14;
  arg1[1]=w2;
  res1[0]=w15;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #35: @17 = (2.*@15) */
  for (i=0, rr=w17, cs=w15; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #36: @9 = (@9+@17) */
  for (i=0, rr=w9, cs=w17; i<6; ++i) (*rr++) += (*cs++);
  /* #37: @18 = 0.125 */
  w18 = 1.2500000000000000e-01;
  /* #38: @15 = (@18*@15) */
  for (i=0, rr=w15, cs=w15; i<6; ++i) (*rr++)  = (w18*(*cs++));
  /* #39: @15 = (@3+@15) */
  for (i=0, rr=w15, cr=w3, cs=w15; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #40: @17 = f(@15, @2) */
  arg1[0]=w15;
  arg1[1]=w2;
  res1[0]=w17;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #41: @9 = (@9+@17) */
  for (i=0, rr=w9, cs=w17; i<6; ++i) (*rr++) += (*cs++);
  /* #42: @9 = (@11*@9) */
  for (i=0, rr=w9, cs=w9; i<6; ++i) (*rr++)  = (w11*(*cs++));
  /* #43: @9 = (@3+@9) */
  for (i=0, rr=w9, cr=w3, cs=w9; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #44: output[0][0] = @9 */
  casadi_copy(w9, 6, res[0]);
  /* #45: @19 = zeros(9x6,15nz) */
  casadi_fill(w19, 15, 0.);
  /* #46: @20 = ones(9x1,3nz) */
  casadi_fill(w20, 3, 1.);
  /* #47: {@21, NULL} = vertsplit(@20) */
  casadi_copy(w20, 3, w21);
  /* #48: @9 = dense(@21) */
  casadi_project(w21, casadi_s1, w9, casadi_s0, w);
  /* #49: @22 = zeros(6x1,0nz) */
  /* #50: @20 = ones(9x1,3nz) */
  casadi_fill(w20, 3, 1.);
  /* #51: {@23, NULL} = vertsplit(@20) */
  casadi_copy(w20, 3, w23);
  /* #52: @24 = zeros(6x1,0nz) */
  /* #53: @17 = horzcat(@21, @23, @24) */
  rr=w17;
  for (i=0, cs=w21; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w23; i<3; ++i) *rr++ = *cs++;
  /* #54: @25 = dense(@17) */
  casadi_project(w17, casadi_s3, w25, casadi_s2, w);
  /* #55: @24 = zeros(3x1,0nz) */
  /* #56: @26 = zeros(3x1,0nz) */
  /* #57: @20 = ones(9x1,3nz) */
  casadi_fill(w20, 3, 1.);
  /* #58: {NULL, @27} = vertsplit(@20) */
  casadi_copy(w20, 3, w27);
  /* #59: @20 = horzcat(@24, @26, @27) */
  rr=w20;
  for (i=0, cs=w27; i<3; ++i) *rr++ = *cs++;
  /* #60: @28 = dense(@20) */
  casadi_project(w20, casadi_s5, w28, casadi_s4, w);
  /* #61: @29 = fwd3_f(@0, @2, @22, @25, @28) */
  arg1[0]=w0;
  arg1[1]=w2;
  arg1[2]=0;
  arg1[3]=w25;
  arg1[4]=w28;
  res1[0]=w29;
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #62: {@0, @17, @30} = horzsplit(@29) */
  casadi_copy(w29, 6, w0);
  casadi_copy(w29+6, 6, w17);
  casadi_copy(w29+12, 6, w30);
  /* #63: @22 = zeros(6x1,0nz) */
  /* #64: @31 = dense(@21) */
  casadi_project(w21, casadi_s1, w31, casadi_s0, w);
  /* #65: @32 = (@4*@0) */
  for (i=0, rr=w32, cs=w0; i<6; ++i) (*rr++)  = (w4*(*cs++));
  /* #66: @31 = (@31+@32) */
  for (i=0, rr=w31, cs=w32; i<6; ++i) (*rr++) += (*cs++);
  /* #67: @32 = dense(@23) */
  casadi_project(w23, casadi_s6, w32, casadi_s0, w);
  /* #68: @33 = (@4*@17) */
  for (i=0, rr=w33, cs=w17; i<6; ++i) (*rr++)  = (w4*(*cs++));
  /* #69: @32 = (@32+@33) */
  for (i=0, rr=w32, cs=w33; i<6; ++i) (*rr++) += (*cs++);
  /* #70: @33 = (@4*@30) */
  for (i=0, rr=w33, cs=w30; i<6; ++i) (*rr++)  = (w4*(*cs++));
  /* #71: @29 = horzcat(@31, @32, @33) */
  rr=w29;
  for (i=0, cs=w31; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w32; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w33; i<6; ++i) *rr++ = *cs++;
  /* #72: @20 = horzcat(@24, @26, @27) */
  rr=w20;
  for (i=0, cs=w27; i<3; ++i) *rr++ = *cs++;
  /* #73: @28 = dense(@20) */
  casadi_project(w20, casadi_s5, w28, casadi_s4, w);
  /* #74: @25 = fwd3_f(@5, @2, @22, @29, @28) */
  arg1[0]=w5;
  arg1[1]=w2;
  arg1[2]=0;
  arg1[3]=w29;
  arg1[4]=w28;
  res1[0]=w25;
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #75: {@5, @31, @32} = horzsplit(@25) */
  casadi_copy(w25, 6, w5);
  casadi_copy(w25+6, 6, w31);
  casadi_copy(w25+12, 6, w32);
  /* #76: @33 = (2.*@5) */
  for (i=0, rr=w33, cs=w5; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #77: @0 = (@0+@33) */
  for (i=0, rr=w0, cs=w33; i<6; ++i) (*rr++) += (*cs++);
  /* #78: @22 = zeros(6x1,0nz) */
  /* #79: @33 = dense(@21) */
  casadi_project(w21, casadi_s1, w33, casadi_s0, w);
  /* #80: @5 = (@8*@5) */
  for (i=0, rr=w5, cs=w5; i<6; ++i) (*rr++)  = (w8*(*cs++));
  /* #81: @33 = (@33+@5) */
  for (i=0, rr=w33, cs=w5; i<6; ++i) (*rr++) += (*cs++);
  /* #82: @5 = dense(@23) */
  casadi_project(w23, casadi_s6, w5, casadi_s0, w);
  /* #83: @34 = (@8*@31) */
  for (i=0, rr=w34, cs=w31; i<6; ++i) (*rr++)  = (w8*(*cs++));
  /* #84: @5 = (@5+@34) */
  for (i=0, rr=w5, cs=w34; i<6; ++i) (*rr++) += (*cs++);
  /* #85: @34 = (@8*@32) */
  for (i=0, rr=w34, cs=w32; i<6; ++i) (*rr++)  = (w8*(*cs++));
  /* #86: @25 = horzcat(@33, @5, @34) */
  rr=w25;
  for (i=0, cs=w33; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w5; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w34; i<6; ++i) *rr++ = *cs++;
  /* #87: @20 = horzcat(@24, @26, @27) */
  rr=w20;
  for (i=0, cs=w27; i<3; ++i) *rr++ = *cs++;
  /* #88: @28 = dense(@20) */
  casadi_project(w20, casadi_s5, w28, casadi_s4, w);
  /* #89: @29 = fwd3_f(@6, @2, @22, @25, @28) */
  arg1[0]=w6;
  arg1[1]=w2;
  arg1[2]=0;
  arg1[3]=w25;
  arg1[4]=w28;
  res1[0]=w29;
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #90: {@6, @33, @5} = horzsplit(@29) */
  casadi_copy(w29, 6, w6);
  casadi_copy(w29+6, 6, w33);
  casadi_copy(w29+12, 6, w5);
  /* #91: @34 = (2.*@6) */
  for (i=0, rr=w34, cs=w6; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #92: @0 = (@0+@34) */
  for (i=0, rr=w0, cs=w34; i<6; ++i) (*rr++) += (*cs++);
  /* #93: @22 = zeros(6x1,0nz) */
  /* #94: @34 = dense(@21) */
  casadi_project(w21, casadi_s1, w34, casadi_s0, w);
  /* #95: @6 = (@10*@6) */
  for (i=0, rr=w6, cs=w6; i<6; ++i) (*rr++)  = (w10*(*cs++));
  /* #96: @34 = (@34+@6) */
  for (i=0, rr=w34, cs=w6; i<6; ++i) (*rr++) += (*cs++);
  /* #97: @6 = dense(@23) */
  casadi_project(w23, casadi_s6, w6, casadi_s0, w);
  /* #98: @35 = (@10*@33) */
  for (i=0, rr=w35, cs=w33; i<6; ++i) (*rr++)  = (w10*(*cs++));
  /* #99: @6 = (@6+@35) */
  for (i=0, rr=w6, cs=w35; i<6; ++i) (*rr++) += (*cs++);
  /* #100: @35 = (@10*@5) */
  for (i=0, rr=w35, cs=w5; i<6; ++i) (*rr++)  = (w10*(*cs++));
  /* #101: @29 = horzcat(@34, @6, @35) */
  rr=w29;
  for (i=0, cs=w34; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w6; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w35; i<6; ++i) *rr++ = *cs++;
  /* #102: @21 = horzcat(@24, @26, @27) */
  rr=w21;
  for (i=0, cs=w27; i<3; ++i) *rr++ = *cs++;
  /* #103: @28 = dense(@21) */
  casadi_project(w21, casadi_s5, w28, casadi_s4, w);
  /* #104: @25 = fwd3_f(@7, @2, @22, @29, @28) */
  arg1[0]=w7;
  arg1[1]=w2;
  arg1[2]=0;
  arg1[3]=w29;
  arg1[4]=w28;
  res1[0]=w25;
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #105: {@7, @34, @6} = horzsplit(@25) */
  casadi_copy(w25, 6, w7);
  casadi_copy(w25+6, 6, w34);
  casadi_copy(w25+12, 6, w6);
  /* #106: @0 = (@0+@7) */
  for (i=0, rr=w0, cs=w7; i<6; ++i) (*rr++) += (*cs++);
  /* #107: @0 = (@1*@0) */
  for (i=0, rr=w0, cs=w0; i<6; ++i) (*rr++)  = (w1*(*cs++));
  /* #108: @9 = (@9+@0) */
  for (i=0, rr=w9, cs=w0; i<6; ++i) (*rr++) += (*cs++);
  /* #109: @22 = zeros(6x1,0nz) */
  /* #110: @0 = dense(@23) */
  casadi_project(w23, casadi_s6, w0, casadi_s0, w);
  /* #111: @31 = (2.*@31) */
  for (i=0, rr=w31, cs=w31; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #112: @17 = (@17+@31) */
  for (i=0, rr=w17, cs=w31; i<6; ++i) (*rr++) += (*cs++);
  /* #113: @33 = (2.*@33) */
  for (i=0, rr=w33, cs=w33; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #114: @17 = (@17+@33) */
  for (i=0, rr=w17, cs=w33; i<6; ++i) (*rr++) += (*cs++);
  /* #115: @17 = (@17+@34) */
  for (i=0, rr=w17, cs=w34; i<6; ++i) (*rr++) += (*cs++);
  /* #116: @17 = (@1*@17) */
  for (i=0, rr=w17, cs=w17; i<6; ++i) (*rr++)  = (w1*(*cs++));
  /* #117: @0 = (@0+@17) */
  for (i=0, rr=w0, cs=w17; i<6; ++i) (*rr++) += (*cs++);
  /* #118: @32 = (2.*@32) */
  for (i=0, rr=w32, cs=w32; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #119: @30 = (@30+@32) */
  for (i=0, rr=w30, cs=w32; i<6; ++i) (*rr++) += (*cs++);
  /* #120: @5 = (2.*@5) */
  for (i=0, rr=w5, cs=w5; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #121: @30 = (@30+@5) */
  for (i=0, rr=w30, cs=w5; i<6; ++i) (*rr++) += (*cs++);
  /* #122: @30 = (@30+@6) */
  for (i=0, rr=w30, cs=w6; i<6; ++i) (*rr++) += (*cs++);
  /* #123: @30 = (@1*@30) */
  for (i=0, rr=w30, cs=w30; i<6; ++i) (*rr++)  = (w1*(*cs++));
  /* #124: @25 = horzcat(@9, @0, @30) */
  rr=w25;
  for (i=0, cs=w9; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w0; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w30; i<6; ++i) *rr++ = *cs++;
  /* #125: @23 = horzcat(@24, @26, @27) */
  rr=w23;
  for (i=0, cs=w27; i<3; ++i) *rr++ = *cs++;
  /* #126: @28 = dense(@23) */
  casadi_project(w23, casadi_s5, w28, casadi_s4, w);
  /* #127: @29 = fwd3_f(@3, @2, @22, @25, @28) */
  arg1[0]=w3;
  arg1[1]=w2;
  arg1[2]=0;
  arg1[3]=w25;
  arg1[4]=w28;
  res1[0]=w29;
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #128: {@3, @6, @5} = horzsplit(@29) */
  casadi_copy(w29, 6, w3);
  casadi_copy(w29+6, 6, w6);
  casadi_copy(w29+12, 6, w5);
  /* #129: @22 = zeros(6x1,0nz) */
  /* #130: @32 = (@12*@3) */
  for (i=0, rr=w32, cs=w3; i<6; ++i) (*rr++)  = (w12*(*cs++));
  /* #131: @32 = (@9+@32) */
  for (i=0, rr=w32, cr=w9, cs=w32; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #132: @17 = (@12*@6) */
  for (i=0, rr=w17, cs=w6; i<6; ++i) (*rr++)  = (w12*(*cs++));
  /* #133: @17 = (@0+@17) */
  for (i=0, rr=w17, cr=w0, cs=w17; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #134: @34 = (@12*@5) */
  for (i=0, rr=w34, cs=w5; i<6; ++i) (*rr++)  = (w12*(*cs++));
  /* #135: @34 = (@30+@34) */
  for (i=0, rr=w34, cr=w30, cs=w34; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #136: @29 = horzcat(@32, @17, @34) */
  rr=w29;
  for (i=0, cs=w32; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w17; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w34; i<6; ++i) *rr++ = *cs++;
  /* #137: @23 = horzcat(@24, @26, @27) */
  rr=w23;
  for (i=0, cs=w27; i<3; ++i) *rr++ = *cs++;
  /* #138: @28 = dense(@23) */
  casadi_project(w23, casadi_s5, w28, casadi_s4, w);
  /* #139: @25 = fwd3_f(@13, @2, @22, @29, @28) */
  arg1[0]=w13;
  arg1[1]=w2;
  arg1[2]=0;
  arg1[3]=w29;
  arg1[4]=w28;
  res1[0]=w25;
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #140: {@13, @32, @17} = horzsplit(@25) */
  casadi_copy(w25, 6, w13);
  casadi_copy(w25+6, 6, w32);
  casadi_copy(w25+12, 6, w17);
  /* #141: @34 = (2.*@13) */
  for (i=0, rr=w34, cs=w13; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #142: @3 = (@3+@34) */
  for (i=0, rr=w3, cs=w34; i<6; ++i) (*rr++) += (*cs++);
  /* #143: @22 = zeros(6x1,0nz) */
  /* #144: @13 = (@16*@13) */
  for (i=0, rr=w13, cs=w13; i<6; ++i) (*rr++)  = (w16*(*cs++));
  /* #145: @13 = (@9+@13) */
  for (i=0, rr=w13, cr=w9, cs=w13; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #146: @34 = (@16*@32) */
  for (i=0, rr=w34, cs=w32; i<6; ++i) (*rr++)  = (w16*(*cs++));
  /* #147: @34 = (@0+@34) */
  for (i=0, rr=w34, cr=w0, cs=w34; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #148: @33 = (@16*@17) */
  for (i=0, rr=w33, cs=w17; i<6; ++i) (*rr++)  = (w16*(*cs++));
  /* #149: @33 = (@30+@33) */
  for (i=0, rr=w33, cr=w30, cs=w33; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #150: @25 = horzcat(@13, @34, @33) */
  rr=w25;
  for (i=0, cs=w13; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w34; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w33; i<6; ++i) *rr++ = *cs++;
  /* #151: @23 = horzcat(@24, @26, @27) */
  rr=w23;
  for (i=0, cs=w27; i<3; ++i) *rr++ = *cs++;
  /* #152: @28 = dense(@23) */
  casadi_project(w23, casadi_s5, w28, casadi_s4, w);
  /* #153: @29 = fwd3_f(@14, @2, @22, @25, @28) */
  arg1[0]=w14;
  arg1[1]=w2;
  arg1[2]=0;
  arg1[3]=w25;
  arg1[4]=w28;
  res1[0]=w29;
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #154: {@14, @13, @34} = horzsplit(@29) */
  casadi_copy(w29, 6, w14);
  casadi_copy(w29+6, 6, w13);
  casadi_copy(w29+12, 6, w34);
  /* #155: @33 = (2.*@14) */
  for (i=0, rr=w33, cs=w14; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #156: @3 = (@3+@33) */
  for (i=0, rr=w3, cs=w33; i<6; ++i) (*rr++) += (*cs++);
  /* #157: @22 = zeros(6x1,0nz) */
  /* #158: @14 = (@18*@14) */
  for (i=0, rr=w14, cs=w14; i<6; ++i) (*rr++)  = (w18*(*cs++));
  /* #159: @14 = (@9+@14) */
  for (i=0, rr=w14, cr=w9, cs=w14; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #160: @33 = (@18*@13) */
  for (i=0, rr=w33, cs=w13; i<6; ++i) (*rr++)  = (w18*(*cs++));
  /* #161: @33 = (@0+@33) */
  for (i=0, rr=w33, cr=w0, cs=w33; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #162: @31 = (@18*@34) */
  for (i=0, rr=w31, cs=w34; i<6; ++i) (*rr++)  = (w18*(*cs++));
  /* #163: @31 = (@30+@31) */
  for (i=0, rr=w31, cr=w30, cs=w31; i<6; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #164: @29 = horzcat(@14, @33, @31) */
  rr=w29;
  for (i=0, cs=w14; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w33; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w31; i<6; ++i) *rr++ = *cs++;
  /* #165: @23 = horzcat(@24, @26, @27) */
  rr=w23;
  for (i=0, cs=w27; i<3; ++i) *rr++ = *cs++;
  /* #166: @28 = dense(@23) */
  casadi_project(w23, casadi_s5, w28, casadi_s4, w);
  /* #167: @25 = fwd3_f(@15, @2, @22, @29, @28) */
  arg1[0]=w15;
  arg1[1]=w2;
  arg1[2]=0;
  arg1[3]=w29;
  arg1[4]=w28;
  res1[0]=w25;
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #168: {@15, @14, @33} = horzsplit(@25) */
  casadi_copy(w25, 6, w15);
  casadi_copy(w25+6, 6, w14);
  casadi_copy(w25+12, 6, w33);
  /* #169: @3 = (@3+@15) */
  for (i=0, rr=w3, cs=w15; i<6; ++i) (*rr++) += (*cs++);
  /* #170: @3 = (@11*@3) */
  for (i=0, rr=w3, cs=w3; i<6; ++i) (*rr++)  = (w11*(*cs++));
  /* #171: @9 = (@9+@3) */
  for (i=0, rr=w9, cs=w3; i<6; ++i) (*rr++) += (*cs++);
  /* #172: @2 = @9[:3] */
  for (rr=w2, ss=w9+0; ss!=w9+3; ss+=1) *rr++ = *ss;
  /* #173: (@19[:9:3] = @2) */
  for (rr=w19+0, ss=w2; rr!=w19+9; rr+=3) *rr = *ss++;
  /* #174: @32 = (2.*@32) */
  for (i=0, rr=w32, cs=w32; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #175: @6 = (@6+@32) */
  for (i=0, rr=w6, cs=w32; i<6; ++i) (*rr++) += (*cs++);
  /* #176: @13 = (2.*@13) */
  for (i=0, rr=w13, cs=w13; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #177: @6 = (@6+@13) */
  for (i=0, rr=w6, cs=w13; i<6; ++i) (*rr++) += (*cs++);
  /* #178: @6 = (@6+@14) */
  for (i=0, rr=w6, cs=w14; i<6; ++i) (*rr++) += (*cs++);
  /* #179: @6 = (@11*@6) */
  for (i=0, rr=w6, cs=w6; i<6; ++i) (*rr++)  = (w11*(*cs++));
  /* #180: @0 = (@0+@6) */
  for (i=0, rr=w0, cs=w6; i<6; ++i) (*rr++) += (*cs++);
  /* #181: (@19[1, 4, 7, 9, 11, 13] = @0) */
  for (cii=casadi_s7, rr=w19, ss=w0; cii!=casadi_s7+6; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #182: @17 = (2.*@17) */
  for (i=0, rr=w17, cs=w17; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #183: @5 = (@5+@17) */
  for (i=0, rr=w5, cs=w17; i<6; ++i) (*rr++) += (*cs++);
  /* #184: @34 = (2.*@34) */
  for (i=0, rr=w34, cs=w34; i<6; ++i) *rr++ = (2.* *cs++ );
  /* #185: @5 = (@5+@34) */
  for (i=0, rr=w5, cs=w34; i<6; ++i) (*rr++) += (*cs++);
  /* #186: @5 = (@5+@33) */
  for (i=0, rr=w5, cs=w33; i<6; ++i) (*rr++) += (*cs++);
  /* #187: @5 = (@11*@5) */
  for (i=0, rr=w5, cs=w5; i<6; ++i) (*rr++)  = (w11*(*cs++));
  /* #188: @30 = (@30+@5) */
  for (i=0, rr=w30, cs=w5; i<6; ++i) (*rr++) += (*cs++);
  /* #189: (@19[2, 5, 8, 10, 12, 14] = @30) */
  for (cii=casadi_s8, rr=w19, ss=w30; cii!=casadi_s8+6; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #190: @36 = @19' */
  casadi_trans(w19,casadi_s10, w36, casadi_s9, iw);
  /* #191: output[1][0] = @36 */
  casadi_copy(w36, 15, res[1]);
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_erk4_chain_nm2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void casadi_erk4_chain_nm2_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_erk4_chain_nm2_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_erk4_chain_nm2_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int casadi_erk4_chain_nm2_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT const char* casadi_erk4_chain_nm2_name_in(casadi_int i){
  switch (i) {
    case 0: return "x0";
    case 1: return "p";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_erk4_chain_nm2_name_out(casadi_int i){
  switch (i) {
    case 0: return "xf";
    case 1: return "sensxu";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_erk4_chain_nm2_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s11;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_erk4_chain_nm2_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s9;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_erk4_chain_nm2_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 10;
  if (sz_w) *sz_w = 200;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
