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
  #define CASADI_PREFIX(ID) engine_impl_dae_jac_x_xdot_u_z_ ## ID
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
#define casadi_fill CASADI_PREFIX(fill)
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
#define casadi_sq CASADI_PREFIX(sq)
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

static const casadi_int casadi_s0[4] = {0, 2, 5, 8};
static const casadi_int casadi_s1[4] = {1, 3, 6, 9};
static const casadi_int casadi_s2[19] = {4, 6, 0, 0, 0, 2, 4, 7, 10, 2, 3, 2, 3, 0, 2, 3, 1, 2, 3};
static const casadi_int casadi_s3[17] = {6, 4, 0, 1, 2, 6, 10, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5};
static const casadi_int casadi_s4[13] = {4, 6, 0, 1, 2, 3, 4, 4, 4, 0, 1, 2, 3};
static const casadi_int casadi_s5[11] = {6, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3};
static const casadi_int casadi_s6[11] = {2, 6, 0, 1, 2, 2, 2, 2, 2, 0, 1};
static const casadi_int casadi_s7[7] = {6, 2, 0, 1, 2, 0, 1};
static const casadi_int casadi_s8[15] = {2, 6, 0, 0, 0, 1, 3, 4, 6, 0, 0, 1, 0, 0, 1};
static const casadi_int casadi_s9[11] = {6, 2, 0, 4, 6, 2, 3, 4, 5, 3, 5};
static const casadi_int casadi_s10[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s11[6] = {2, 1, 0, 2, 0, 1};

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

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

casadi_real casadi_sq(casadi_real x) { return x*x;}

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

/* engine_impl_dae_jac_x_xdot_u_z:(i0[4],i1[4],i2[2],i3[2])->(o0[6x4,10nz],o1[6x4,4nz],o2[6x2,2nz],o3[6x2,6nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real *rr, *ss, *tt;
  const casadi_int *cii;
  const casadi_real *cs;
  casadi_real *w0=w+0, w2, w3, w4, *w5=w+13, w6, w7, w8, w9, w10, w11, w12, w13, w14, *w15=w+24, *w16=w+28, *w17=w+30, w18, w19, w20, *w21=w+35, *w22=w+37, w23, w24, w25, w26, w27, w28, w29, w30, w31, w32, w33, w34, w35, w36, w37, w38, w40, w41, w42, w43, w44, w45, w46, w47, *w48=w+63, *w49=w+67, *w50=w+77, *w52=w+83;
  /* #0: @0 = zeros(4x6,10nz) */
  casadi_fill(w0, 10, 0.);
  /* #1: @1 = zeros(4x1,0nz) */
  /* #2: @2 = 3.85 */
  w2 = 3.8500000000000001e+00;
  /* #3: @3 = 0.0001 */
  w3 = 1.0000000000000000e-04;
  /* #4: @4 = 0.0001 */
  w4 = 1.0000000000000000e-04;
  /* #5: @5 = input[3][0] */
  casadi_copy(arg[3], 2, w5);
  /* #6: @6 = @5[0] */
  for (rr=(&w6), ss=w5+0; ss!=w5+1; ss+=1) *rr++ = *ss;
  /* #7: @7 = sq(@6) */
  w7 = casadi_sq( w6 );
  /* #8: @4 = (@4+@7) */
  w4 += w7;
  /* #9: @4 = sqrt(@4) */
  w4 = sqrt( w4 );
  /* #10: @7 = (@4+@6) */
  w7  = (w4+w6);
  /* #11: @8 = 2 */
  w8 = 2.;
  /* #12: @7 = (@7/@8) */
  w7 /= w8;
  /* #13: @8 = 0.5 */
  w8 = 5.0000000000000000e-01;
  /* #14: @9 = pow(@7,@8) */
  w9  = pow(w7,w8);
  /* #15: @10 = 0.25 */
  w10 = 2.5000000000000000e-01;
  /* #16: @11 = pow(@7,@10) */
  w11  = pow(w7,w10);
  /* #17: @9 = (@9-@11) */
  w9 -= w11;
  /* #18: @11 = sq(@9) */
  w11 = casadi_sq( w9 );
  /* #19: @3 = (@3+@11) */
  w3 += w11;
  /* #20: @3 = sqrt(@3) */
  w3 = sqrt( w3 );
  /* #21: @11 = (@3+@9) */
  w11  = (w3+w9);
  /* #22: @12 = 2 */
  w12 = 2.;
  /* #23: @11 = (@11/@12) */
  w11 /= w12;
  /* #24: @11 = sqrt(@11) */
  w11 = sqrt( w11 );
  /* #25: @12 = (@2*@11) */
  w12  = (w2*w11);
  /* #26: @13 = 0.6 */
  w13 = 5.9999999999999998e-01;
  /* #27: @14 = 1 */
  w14 = 1.;
  /* #28: @15 = input[0][0] */
  casadi_copy(arg[0], 4, w15);
  /* #29: {@16, @17} = vertsplit(@15) */
  casadi_copy(w15, 2, w16);
  casadi_copy(w15+2, 2, w17);
  /* #30: @18 = @16[0] */
  for (rr=(&w18), ss=w16+0; ss!=w16+1; ss+=1) *rr++ = *ss;
  /* #31: @19 = 67.5 */
  w19 = 6.7500015417956192e+01;
  /* #32: @18 = (@18-@19) */
  w18 -= w19;
  /* #33: @19 = 4.71181 */
  w19 = 4.7118112699096235e+00;
  /* #34: @18 = (@18/@19) */
  w18 /= w19;
  /* #35: @18 = (-@18) */
  w18 = (- w18 );
  /* #36: @18 = exp(@18) */
  w18 = exp( w18 );
  /* #37: @14 = (@14+@18) */
  w14 += w18;
  /* #38: @19 = (1./@14) */
  w19 = (1./ w14 );
  /* #39: @14 = (@19/@14) */
  w14  = (w19/w14);
  /* #40: @20 = 0.212233 */
  w20 = 2.1223260922740245e-01;
  /* #41: @21 = ones(4x1,2nz) */
  casadi_fill(w21, 2, 1.);
  /* #42: {@22, NULL} = vertsplit(@21) */
  casadi_copy(w21, 2, w22);
  /* #43: @23 = @22[0] */
  for (rr=(&w23), ss=w22+0; ss!=w22+1; ss+=1) *rr++ = *ss;
  /* #44: @20 = (@20*@23) */
  w20 *= w23;
  /* #45: @18 = (@18*@20) */
  w18 *= w20;
  /* #46: @14 = (@14*@18) */
  w14 *= w18;
  /* #47: @18 = 1 */
  w18 = 1.;
  /* #48: @20 = @17[0] */
  for (rr=(&w20), ss=w17+0; ss!=w17+1; ss+=1) *rr++ = *ss;
  /* #49: @23 = @17[1] */
  for (rr=(&w23), ss=w17+1; ss!=w17+2; ss+=1) *rr++ = *ss;
  /* #50: @24 = (@20*@23) */
  w24  = (w20*w23);
  /* #51: @25 = 1.49 */
  w25 = 1.4900000808994172e+00;
  /* #52: @24 = (@24-@25) */
  w24 -= w25;
  /* #53: @25 = 0.037675 */
  w25 = 3.7675016210209952e-02;
  /* #54: @24 = (@24/@25) */
  w24 /= w25;
  /* #55: @24 = (-@24) */
  w24 = (- w24 );
  /* #56: @24 = exp(@24) */
  w24 = exp( w24 );
  /* #57: @18 = (@18+@24) */
  w18 += w24;
  /* #58: @14 = (@14/@18) */
  w14 /= w18;
  /* #59: @14 = (@13*@14) */
  w14  = (w13*w14);
  /* #60: @14 = (@12*@14) */
  w14  = (w12*w14);
  /* #61: @14 = (-@14) */
  w14 = (- w14 );
  /* #62: @25 = 1.8 */
  w25 = 1.8000000000000000e+00;
  /* #63: @26 = (@25*@7) */
  w26  = (w25*w7);
  /* #64: @27 = 0.0001 */
  w27 = 1.0000000000000000e-04;
  /* #65: @28 = 0.0001 */
  w28 = 1.0000000000000000e-04;
  /* #66: @29 = @5[1] */
  for (rr=(&w29), ss=w5+1; ss!=w5+2; ss+=1) *rr++ = *ss;
  /* #67: @30 = sq(@29) */
  w30 = casadi_sq( w29 );
  /* #68: @28 = (@28+@30) */
  w28 += w30;
  /* #69: @28 = sqrt(@28) */
  w28 = sqrt( w28 );
  /* #70: @30 = (@28+@29) */
  w30  = (w28+w29);
  /* #71: @31 = 2 */
  w31 = 2.;
  /* #72: @30 = (@30/@31) */
  w30 /= w31;
  /* #73: @31 = 0.5 */
  w31 = 5.0000000000000000e-01;
  /* #74: @32 = pow(@30,@31) */
  w32  = pow(w30,w31);
  /* #75: @33 = 0.25 */
  w33 = 2.5000000000000000e-01;
  /* #76: @34 = pow(@30,@33) */
  w34  = pow(w30,w33);
  /* #77: @32 = (@32-@34) */
  w32 -= w34;
  /* #78: @34 = sq(@32) */
  w34 = casadi_sq( w32 );
  /* #79: @27 = (@27+@34) */
  w27 += w34;
  /* #80: @27 = sqrt(@27) */
  w27 = sqrt( w27 );
  /* #81: @34 = (@27+@32) */
  w34  = (w27+w32);
  /* #82: @35 = 2 */
  w35 = 2.;
  /* #83: @34 = (@34/@35) */
  w34 /= w35;
  /* #84: @34 = sqrt(@34) */
  w34 = sqrt( w34 );
  /* #85: @35 = (@26*@34) */
  w35  = (w26*w34);
  /* #86: @36 = 0.9 */
  w36 = 9.0000000000000002e-01;
  /* #87: @37 = 0.01 */
  w37 = 1.0000000000000000e-02;
  /* #88: @38 = @22[1] */
  for (rr=(&w38), ss=w22+1; ss!=w22+2; ss+=1) *rr++ = *ss;
  /* #89: @37 = (@37*@38) */
  w37 *= w38;
  /* #90: @37 = (@36*@37) */
  w37  = (w36*w37);
  /* #91: @37 = (@35*@37) */
  w37  = (w35*w37);
  /* #92: @37 = (-@37) */
  w37 = (- w37 );
  /* #93: @22 = vertcat(@1, @14, @37) */
  rr=w22;
  *rr++ = w14;
  *rr++ = w37;
  /* #94: @5 = @22[:2] */
  for (rr=w5, ss=w22+0; ss!=w22+2; ss+=1) *rr++ = *ss;
  /* #95: (@0[4:10:3] = @5) */
  for (rr=w0+4, ss=w5; rr!=w0+10; rr+=3) *rr = *ss++;
  /* #96: @1 = 00 */
  /* #97: @39 = 00 */
  /* #98: @14 = 6.8 */
  w14 = 6.7999999999999998e+00;
  /* #99: @37 = (@14*@23) */
  w37  = (w14*w23);
  /* #100: @38 = 1.29 */
  w38 = 1.2900000000000000e+00;
  /* #101: @40 = 0.0001 */
  w40 = 1.0000000000000000e-04;
  /* #102: @41 = sq(@20) */
  w41 = casadi_sq( w20 );
  /* #103: @40 = (@40+@41) */
  w40 += w41;
  /* #104: @40 = sqrt(@40) */
  w40 = sqrt( w40 );
  /* #105: @41 = (@40+@20) */
  w41  = (w40+w20);
  /* #106: @42 = 2 */
  w42 = 2.;
  /* #107: @41 = (@41/@42) */
  w41 /= w42;
  /* #108: @42 = 0.29 */
  w42 = 2.9000000000000004e-01;
  /* #109: @42 = pow(@41,@42) */
  w42  = pow(w41,w42);
  /* #110: @42 = (@38*@42) */
  w42  = (w38*w42);
  /* #111: @43 = 0.5 */
  w43 = 5.0000000000000000e-01;
  /* #112: @44 = (2.*@20) */
  w44 = (2.* w20 );
  /* #113: @45 = ones(4x1,1nz) */
  w45 = 1.;
  /* #114: {NULL, @46} = vertsplit(@45) */
  w46 = w45;
  /* #115: @45 = @46[0] */
  for (rr=(&w45), ss=(&w46)+0; ss!=(&w46)+1; ss+=1) *rr++ = *ss;
  /* #116: @44 = (@44*@45) */
  w44 *= w45;
  /* #117: @40 = (2.*@40) */
  w40 = (2.* w40 );
  /* #118: @44 = (@44/@40) */
  w44 /= w40;
  /* #119: @44 = (@44+@45) */
  w44 += w45;
  /* #120: @43 = (@43*@44) */
  w43 *= w44;
  /* #121: @42 = (@42*@43) */
  w42 *= w43;
  /* #122: @42 = (@42-@45) */
  w42 -= w45;
  /* #123: @37 = (@37*@42) */
  w37 *= w42;
  /* #124: @37 = (-@37) */
  w37 = (- w37 );
  /* #125: @42 = 0.0001 */
  w42 = 1.0000000000000000e-04;
  /* #126: @43 = sq(@23) */
  w43 = casadi_sq( w23 );
  /* #127: @42 = (@42+@43) */
  w42 += w43;
  /* #128: @42 = sqrt(@42) */
  w42 = sqrt( w42 );
  /* #129: @43 = (@42+@23) */
  w43  = (w42+w23);
  /* #130: @44 = 2 */
  w44 = 2.;
  /* #131: @43 = (@43/@44) */
  w43 /= w44;
  /* #132: @44 = 1.29 */
  w44 = 1.2900000000000000e+00;
  /* #133: @40 = pow(@43,@44) */
  w40  = pow(w43,w44);
  /* #134: @40 = (@40-@23) */
  w40 -= w23;
  /* #135: @46 = 18.4 */
  w46 = 1.8399999999999999e+01;
  /* #136: @47 = (@46*@45) */
  w47  = (w46*w45);
  /* #137: @40 = (@40*@47) */
  w40 *= w47;
  /* #138: @40 = (-@40) */
  w40 = (- w40 );
  /* #139: @5 = vertcat(@1, @39, @37, @40) */
  rr=w5;
  *rr++ = w37;
  *rr++ = w40;
  /* #140: @5 = (-@5) */
  for (i=0, rr=w5, cs=w5; i<2; ++i) *rr++ = (- *cs++ );
  /* #141: @37 = 1 */
  w37 = 1.;
  /* #142: @37 = (@37-@19) */
  w37 -= w19;
  /* #143: @37 = (@37/@18) */
  w37 /= w18;
  /* #144: @18 = (@37/@18) */
  w18  = (w37/w18);
  /* #145: @19 = 26.5428 */
  w19 = 2.6542788845011813e+01;
  /* #146: @40 = (@23*@45) */
  w40  = (w23*w45);
  /* #147: @40 = (@19*@40) */
  w40  = (w19*w40);
  /* #148: @40 = (@24*@40) */
  w40  = (w24*w40);
  /* #149: @40 = (@18*@40) */
  w40  = (w18*w40);
  /* #150: @40 = (@13*@40) */
  w40  = (w13*w40);
  /* #151: @40 = (@12*@40) */
  w40  = (w12*w40);
  /* #152: @47 = (@23*@45) */
  w47  = (w23*w45);
  /* #153: @40 = (@40-@47) */
  w40 -= w47;
  /* #154: @45 = (@23*@45) */
  w45  = (w23*w45);
  /* #155: @45 = (-@45) */
  w45 = (- w45 );
  /* #156: @15 = vertcat(@5, @40, @45) */
  rr=w15;
  for (i=0, cs=w5; i<2; ++i) *rr++ = *cs++;
  *rr++ = w40;
  *rr++ = w45;
  /* #157: @48 = @15[:4] */
  for (rr=w48, ss=w15+0; ss!=w15+4; ss+=1) *rr++ = *ss;
  /* #158: (@0[0, 2, 5, 8] = @48) */
  for (cii=casadi_s0, rr=w0, ss=w48; cii!=casadi_s0+4; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #159: @1 = 00 */
  /* #160: @39 = 00 */
  /* #161: @41 = pow(@41,@38) */
  w41  = pow(w41,w38);
  /* #162: @41 = (@41-@20) */
  w41 -= w20;
  /* #163: @38 = ones(4x1,1nz) */
  w38 = 1.;
  /* #164: {NULL, @40} = vertsplit(@38) */
  w40 = w38;
  /* #165: @38 = @40[0] */
  for (rr=(&w38), ss=(&w40)+0; ss!=(&w40)+1; ss+=1) *rr++ = *ss;
  /* #166: @14 = (@14*@38) */
  w14 *= w38;
  /* #167: @41 = (@41*@14) */
  w41 *= w14;
  /* #168: @41 = (-@41) */
  w41 = (- w41 );
  /* #169: @46 = (@46*@20) */
  w46 *= w20;
  /* #170: @14 = 0.29 */
  w14 = 2.9000000000000004e-01;
  /* #171: @43 = pow(@43,@14) */
  w43  = pow(w43,w14);
  /* #172: @44 = (@44*@43) */
  w44 *= w43;
  /* #173: @43 = 0.5 */
  w43 = 5.0000000000000000e-01;
  /* #174: @23 = (2.*@23) */
  w23 = (2.* w23 );
  /* #175: @23 = (@23*@38) */
  w23 *= w38;
  /* #176: @42 = (2.*@42) */
  w42 = (2.* w42 );
  /* #177: @23 = (@23/@42) */
  w23 /= w42;
  /* #178: @23 = (@23+@38) */
  w23 += w38;
  /* #179: @43 = (@43*@23) */
  w43 *= w23;
  /* #180: @44 = (@44*@43) */
  w44 *= w43;
  /* #181: @44 = (@44-@38) */
  w44 -= w38;
  /* #182: @46 = (@46*@44) */
  w46 *= w44;
  /* #183: @46 = (-@46) */
  w46 = (- w46 );
  /* #184: @5 = vertcat(@1, @39, @41, @46) */
  rr=w5;
  *rr++ = w41;
  *rr++ = w46;
  /* #185: @5 = (-@5) */
  for (i=0, rr=w5, cs=w5; i<2; ++i) *rr++ = (- *cs++ );
  /* #186: @41 = (@20*@38) */
  w41  = (w20*w38);
  /* #187: @19 = (@19*@41) */
  w19 *= w41;
  /* #188: @24 = (@24*@19) */
  w24 *= w19;
  /* #189: @18 = (@18*@24) */
  w18 *= w24;
  /* #190: @18 = (@13*@18) */
  w18  = (w13*w18);
  /* #191: @18 = (@12*@18) */
  w18  = (w12*w18);
  /* #192: @24 = (@20*@38) */
  w24  = (w20*w38);
  /* #193: @18 = (@18-@24) */
  w18 -= w24;
  /* #194: @20 = (@20*@38) */
  w20 *= w38;
  /* #195: @20 = (-@20) */
  w20 = (- w20 );
  /* #196: @48 = vertcat(@5, @18, @20) */
  rr=w48;
  for (i=0, cs=w5; i<2; ++i) *rr++ = *cs++;
  *rr++ = w18;
  *rr++ = w20;
  /* #197: @15 = @48[:4] */
  for (rr=w15, ss=w48+0; ss!=w48+4; ss+=1) *rr++ = *ss;
  /* #198: (@0[1, 3, 6, 9] = @15) */
  for (cii=casadi_s1, rr=w0, ss=w15; cii!=casadi_s1+4; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #199: @49 = @0' */
  casadi_trans(w0,casadi_s2, w49, casadi_s3, iw);
  /* #200: output[0][0] = @49 */
  casadi_copy(w49, 10, res[0]);
  /* #201: @15 = zeros(4x6,4nz) */
  casadi_fill(w15, 4, 0.);
  /* #202: @48 = ones(4x1) */
  casadi_fill(w48, 4, 1.);
  /* #203: (@15[:4] = @48) */
  for (rr=w15+0, ss=w48; rr!=w15+4; rr+=1) *rr = *ss++;
  /* #204: @48 = @15' */
  casadi_trans(w15,casadi_s4, w48, casadi_s5, iw);
  /* #205: output[1][0] = @48 */
  casadi_copy(w48, 4, res[1]);
  /* #206: @5 = zeros(2x6,2nz) */
  casadi_fill(w5, 2, 0.);
  /* #207: @18 = 1 */
  w18 = 1.;
  /* #208: @20 = 1 */
  w20 = 1.;
  /* #209: @1 = 00 */
  /* #210: @39 = 00 */
  /* #211: @22 = vertcat(@18, @20, @1, @39) */
  rr=w22;
  *rr++ = w18;
  *rr++ = w20;
  /* #212: @22 = (-@22) */
  for (i=0, rr=w22, cs=w22; i<2; ++i) *rr++ = (- *cs++ );
  /* #213: @17 = @22[:2] */
  for (rr=w17, ss=w22+0; ss!=w22+2; ss+=1) *rr++ = *ss;
  /* #214: (@5[:2] = @17) */
  for (rr=w5+0, ss=w17; rr!=w5+2; rr+=1) *rr = *ss++;
  /* #215: @17 = @5' */
  casadi_trans(w5,casadi_s6, w17, casadi_s7, iw);
  /* #216: output[2][0] = @17 */
  casadi_copy(w17, 2, res[2]);
  /* #217: @50 = zeros(2x6,6nz) */
  casadi_fill(w50, 6, 0.);
  /* #218: @1 = 00 */
  /* #219: @39 = 00 */
  /* #220: @18 = 0.0001 */
  w18 = 1.0000000000000000e-04;
  /* #221: @20 = -1.5 */
  w20 = -1.5000000000000000e+00;
  /* #222: @38 = pow(@7,@20) */
  w38  = pow(w7,w20);
  /* #223: @24 = -1.75 */
  w24 = -1.7500000000000000e+00;
  /* #224: @19 = pow(@7,@24) */
  w19  = pow(w7,w24);
  /* #225: @38 = (@38-@19) */
  w38 -= w19;
  /* #226: @19 = sq(@38) */
  w19 = casadi_sq( w38 );
  /* #227: @18 = (@18+@19) */
  w18 += w19;
  /* #228: @18 = sqrt(@18) */
  w18 = sqrt( w18 );
  /* #229: @19 = (@18+@38) */
  w19  = (w18+w38);
  /* #230: @41 = 2 */
  w41 = 2.;
  /* #231: @19 = (@19/@41) */
  w19 /= w41;
  /* #232: @19 = sqrt(@19) */
  w19 = sqrt( w19 );
  /* #233: @41 = 25.3 */
  w41 = 2.5300000000000001e+01;
  /* #234: @46 = 1.5 */
  w46 = 1.5000000000000000e+00;
  /* #235: @44 = 0.5 */
  w44 = 5.0000000000000000e-01;
  /* #236: @44 = pow(@7,@44) */
  w44  = pow(w7,w44);
  /* #237: @44 = (@46*@44) */
  w44  = (w46*w44);
  /* #238: @43 = 0.5 */
  w43 = 5.0000000000000000e-01;
  /* #239: @23 = 1 */
  w23 = 1.;
  /* #240: @6 = (2.*@6) */
  w6 = (2.* w6 );
  /* #241: @4 = (2.*@4) */
  w4 = (2.* w4 );
  /* #242: @6 = (@6/@4) */
  w6 /= w4;
  /* #243: @23 = (@23+@6) */
  w23 += w6;
  /* #244: @43 = (@43*@23) */
  w43 *= w23;
  /* #245: @44 = (@44*@43) */
  w44 *= w43;
  /* #246: @23 = 1.25 */
  w23 = 1.2500000000000000e+00;
  /* #247: @6 = 0.25 */
  w6 = 2.5000000000000000e-01;
  /* #248: @6 = pow(@7,@6) */
  w6  = pow(w7,w6);
  /* #249: @6 = (@23*@6) */
  w6  = (w23*w6);
  /* #250: @6 = (@6*@43) */
  w6 *= w43;
  /* #251: @44 = (@44-@6) */
  w44 -= w6;
  /* #252: @44 = (@41*@44) */
  w44  = (w41*w44);
  /* #253: @44 = (@19*@44) */
  w44  = (w19*w44);
  /* #254: @46 = pow(@7,@46) */
  w46  = pow(w7,w46);
  /* #255: @23 = pow(@7,@23) */
  w23  = pow(w7,w23);
  /* #256: @46 = (@46-@23) */
  w46 -= w23;
  /* #257: @41 = (@41*@46) */
  w41 *= w46;
  /* #258: @46 = 0.5 */
  w46 = 5.0000000000000000e-01;
  /* #259: @38 = (2.*@38) */
  w38 = (2.* w38 );
  /* #260: @23 = -2.5 */
  w23 = -2.5000000000000000e+00;
  /* #261: @23 = pow(@7,@23) */
  w23  = pow(w7,w23);
  /* #262: @20 = (@20*@23) */
  w20 *= w23;
  /* #263: @20 = (@20*@43) */
  w20 *= w43;
  /* #264: @23 = -2.75 */
  w23 = -2.7500000000000000e+00;
  /* #265: @23 = pow(@7,@23) */
  w23  = pow(w7,w23);
  /* #266: @24 = (@24*@23) */
  w24 *= w23;
  /* #267: @24 = (@24*@43) */
  w24 *= w43;
  /* #268: @20 = (@20-@24) */
  w20 -= w24;
  /* #269: @38 = (@38*@20) */
  w38 *= w20;
  /* #270: @18 = (2.*@18) */
  w18 = (2.* w18 );
  /* #271: @38 = (@38/@18) */
  w38 /= w18;
  /* #272: @38 = (@38+@20) */
  w38 += w20;
  /* #273: @46 = (@46*@38) */
  w46 *= w38;
  /* #274: @19 = (2.*@19) */
  w19 = (2.* w19 );
  /* #275: @46 = (@46/@19) */
  w46 /= w19;
  /* #276: @41 = (@41*@46) */
  w41 *= w46;
  /* #277: @44 = (@44+@41) */
  w44 += w41;
  /* #278: @41 = 0.0001 */
  w41 = 1.0000000000000000e-04;
  /* #279: @46 = -1.5 */
  w46 = -1.5000000000000000e+00;
  /* #280: @19 = pow(@30,@46) */
  w19  = pow(w30,w46);
  /* #281: @38 = -1.75 */
  w38 = -1.7500000000000000e+00;
  /* #282: @20 = pow(@30,@38) */
  w20  = pow(w30,w38);
  /* #283: @19 = (@19-@20) */
  w19 -= w20;
  /* #284: @20 = sq(@19) */
  w20 = casadi_sq( w19 );
  /* #285: @41 = (@41+@20) */
  w41 += w20;
  /* #286: @41 = sqrt(@41) */
  w41 = sqrt( w41 );
  /* #287: @20 = (@41+@19) */
  w20  = (w41+w19);
  /* #288: @18 = 2 */
  w18 = 2.;
  /* #289: @20 = (@20/@18) */
  w20 /= w18;
  /* #290: @20 = sqrt(@20) */
  w20 = sqrt( w20 );
  /* #291: @18 = 1.5 */
  w18 = 1.5000000000000000e+00;
  /* #292: @24 = pow(@30,@18) */
  w24  = pow(w30,w18);
  /* #293: @23 = 1.25 */
  w23 = 1.2500000000000000e+00;
  /* #294: @6 = pow(@30,@23) */
  w6  = pow(w30,w23);
  /* #295: @24 = (@24-@6) */
  w24 -= w6;
  /* #296: @6 = 43.6 */
  w6 = 4.3600000000000001e+01;
  /* #297: @4 = (@6*@43) */
  w4  = (w6*w43);
  /* #298: @4 = (@24*@4) */
  w4  = (w24*w4);
  /* #299: @4 = (@20*@4) */
  w4  = (w20*w4);
  /* #300: @17 = vertcat(@1, @39, @44, @4) */
  rr=w17;
  *rr++ = w44;
  *rr++ = w4;
  /* #301: @17 = (-@17) */
  for (i=0, rr=w17, cs=w17; i<2; ++i) *rr++ = (- *cs++ );
  /* #302: @44 = 0.5 */
  w44 = 5.0000000000000000e-01;
  /* #303: @4 = pow(@7,@44) */
  w4  = pow(w7,w44);
  /* #304: @13 = (@13*@37) */
  w13 *= w37;
  /* #305: @4 = (@4+@13) */
  w4 += w13;
  /* #306: @13 = 0.5 */
  w13 = 5.0000000000000000e-01;
  /* #307: @9 = (2.*@9) */
  w9 = (2.* w9 );
  /* #308: @37 = -0.5 */
  w37 = -5.0000000000000000e-01;
  /* #309: @37 = pow(@7,@37) */
  w37  = pow(w7,w37);
  /* #310: @8 = (@8*@37) */
  w8 *= w37;
  /* #311: @8 = (@8*@43) */
  w8 *= w43;
  /* #312: @37 = -0.75 */
  w37 = -7.5000000000000000e-01;
  /* #313: @37 = pow(@7,@37) */
  w37  = pow(w7,w37);
  /* #314: @10 = (@10*@37) */
  w10 *= w37;
  /* #315: @10 = (@10*@43) */
  w10 *= w43;
  /* #316: @8 = (@8-@10) */
  w8 -= w10;
  /* #317: @9 = (@9*@8) */
  w9 *= w8;
  /* #318: @3 = (2.*@3) */
  w3 = (2.* w3 );
  /* #319: @9 = (@9/@3) */
  w9 /= w3;
  /* #320: @9 = (@9+@8) */
  w9 += w8;
  /* #321: @13 = (@13*@9) */
  w13 *= w9;
  /* #322: @11 = (2.*@11) */
  w11 = (2.* w11 );
  /* #323: @13 = (@13/@11) */
  w13 /= w11;
  /* #324: @2 = (@2*@13) */
  w2 *= w13;
  /* #325: @4 = (@4*@2) */
  w4 *= w2;
  /* #326: @2 = -0.5 */
  w2 = -5.0000000000000000e-01;
  /* #327: @2 = pow(@7,@2) */
  w2  = pow(w7,w2);
  /* #328: @44 = (@44*@2) */
  w44 *= w2;
  /* #329: @44 = (@44*@43) */
  w44 *= w43;
  /* #330: @12 = (@12*@44) */
  w12 *= w44;
  /* #331: @4 = (@4+@12) */
  w4 += w12;
  /* #332: @12 = 0.5 */
  w12 = 5.0000000000000000e-01;
  /* #333: @44 = pow(@30,@12) */
  w44  = pow(w30,w12);
  /* #334: @2 = 1 */
  w2 = 1.;
  /* #335: @13 = @16[1] */
  for (rr=(&w13), ss=w16+1; ss!=w16+2; ss+=1) *rr++ = *ss;
  /* #336: @11 = 100 */
  w11 = 100.;
  /* #337: @13 = (@13/@11) */
  w13 /= w11;
  /* #338: @2 = (@2-@13) */
  w2 -= w13;
  /* #339: @36 = (@36*@2) */
  w36 *= w2;
  /* #340: @44 = (@44+@36) */
  w44 += w36;
  /* #341: @25 = (@25*@43) */
  w25 *= w43;
  /* #342: @25 = (@34*@25) */
  w25  = (w34*w25);
  /* #343: @25 = (@44*@25) */
  w25  = (w44*w25);
  /* #344: @48 = vertcat(@17, @4, @25) */
  rr=w48;
  for (i=0, cs=w17; i<2; ++i) *rr++ = *cs++;
  *rr++ = w4;
  *rr++ = w25;
  /* #345: @15 = @48[:4] */
  for (rr=w15, ss=w48+0; ss!=w48+4; ss+=1) *rr++ = *ss;
  /* #346: (@50[:6:3;:2] = @15) */
  for (rr=w50+0, ss=w15; rr!=w50+6; rr+=3) for (tt=rr+0; tt!=rr+2; tt+=1) *tt = *ss++;
  /* #347: @1 = 00 */
  /* #348: @39 = 00 */
  /* #349: @51 = 00 */
  /* #350: @6 = (@6*@7) */
  w6 *= w7;
  /* #351: @7 = 0.5 */
  w7 = 5.0000000000000000e-01;
  /* #352: @7 = pow(@30,@7) */
  w7  = pow(w30,w7);
  /* #353: @18 = (@18*@7) */
  w18 *= w7;
  /* #354: @7 = 0.5 */
  w7 = 5.0000000000000000e-01;
  /* #355: @4 = 1 */
  w4 = 1.;
  /* #356: @29 = (2.*@29) */
  w29 = (2.* w29 );
  /* #357: @28 = (2.*@28) */
  w28 = (2.* w28 );
  /* #358: @29 = (@29/@28) */
  w29 /= w28;
  /* #359: @4 = (@4+@29) */
  w4 += w29;
  /* #360: @7 = (@7*@4) */
  w7 *= w4;
  /* #361: @18 = (@18*@7) */
  w18 *= w7;
  /* #362: @4 = 0.25 */
  w4 = 2.5000000000000000e-01;
  /* #363: @4 = pow(@30,@4) */
  w4  = pow(w30,w4);
  /* #364: @23 = (@23*@4) */
  w23 *= w4;
  /* #365: @23 = (@23*@7) */
  w23 *= w7;
  /* #366: @18 = (@18-@23) */
  w18 -= w23;
  /* #367: @18 = (@6*@18) */
  w18  = (w6*w18);
  /* #368: @18 = (@20*@18) */
  w18  = (w20*w18);
  /* #369: @6 = (@6*@24) */
  w6 *= w24;
  /* #370: @24 = 0.5 */
  w24 = 5.0000000000000000e-01;
  /* #371: @19 = (2.*@19) */
  w19 = (2.* w19 );
  /* #372: @23 = -2.5 */
  w23 = -2.5000000000000000e+00;
  /* #373: @23 = pow(@30,@23) */
  w23  = pow(w30,w23);
  /* #374: @46 = (@46*@23) */
  w46 *= w23;
  /* #375: @46 = (@46*@7) */
  w46 *= w7;
  /* #376: @23 = -2.75 */
  w23 = -2.7500000000000000e+00;
  /* #377: @23 = pow(@30,@23) */
  w23  = pow(w30,w23);
  /* #378: @38 = (@38*@23) */
  w38 *= w23;
  /* #379: @38 = (@38*@7) */
  w38 *= w7;
  /* #380: @46 = (@46-@38) */
  w46 -= w38;
  /* #381: @19 = (@19*@46) */
  w19 *= w46;
  /* #382: @41 = (2.*@41) */
  w41 = (2.* w41 );
  /* #383: @19 = (@19/@41) */
  w19 /= w41;
  /* #384: @19 = (@19+@46) */
  w19 += w46;
  /* #385: @24 = (@24*@19) */
  w24 *= w19;
  /* #386: @20 = (2.*@20) */
  w20 = (2.* w20 );
  /* #387: @24 = (@24/@20) */
  w24 /= w20;
  /* #388: @6 = (@6*@24) */
  w6 *= w24;
  /* #389: @18 = (@18+@6) */
  w18 += w6;
  /* #390: @6 = vertcat(@1, @39, @51, @18) */
  rr=(&w6);
  *rr++ = w18;
  /* #391: @6 = (-@6) */
  w6 = (- w6 );
  /* #392: @1 = 00 */
  /* #393: @18 = 0.5 */
  w18 = 5.0000000000000000e-01;
  /* #394: @32 = (2.*@32) */
  w32 = (2.* w32 );
  /* #395: @24 = -0.5 */
  w24 = -5.0000000000000000e-01;
  /* #396: @24 = pow(@30,@24) */
  w24  = pow(w30,w24);
  /* #397: @31 = (@31*@24) */
  w31 *= w24;
  /* #398: @31 = (@31*@7) */
  w31 *= w7;
  /* #399: @24 = -0.75 */
  w24 = -7.5000000000000000e-01;
  /* #400: @24 = pow(@30,@24) */
  w24  = pow(w30,w24);
  /* #401: @33 = (@33*@24) */
  w33 *= w24;
  /* #402: @33 = (@33*@7) */
  w33 *= w7;
  /* #403: @31 = (@31-@33) */
  w31 -= w33;
  /* #404: @32 = (@32*@31) */
  w32 *= w31;
  /* #405: @27 = (2.*@27) */
  w27 = (2.* w27 );
  /* #406: @32 = (@32/@27) */
  w32 /= w27;
  /* #407: @32 = (@32+@31) */
  w32 += w31;
  /* #408: @18 = (@18*@32) */
  w18 *= w32;
  /* #409: @34 = (2.*@34) */
  w34 = (2.* w34 );
  /* #410: @18 = (@18/@34) */
  w18 /= w34;
  /* #411: @26 = (@26*@18) */
  w26 *= w18;
  /* #412: @44 = (@44*@26) */
  w44 *= w26;
  /* #413: @26 = -0.5 */
  w26 = -5.0000000000000000e-01;
  /* #414: @30 = pow(@30,@26) */
  w30  = pow(w30,w26);
  /* #415: @12 = (@12*@30) */
  w12 *= w30;
  /* #416: @12 = (@12*@7) */
  w12 *= w7;
  /* #417: @35 = (@35*@12) */
  w35 *= w12;
  /* #418: @44 = (@44+@35) */
  w44 += w35;
  /* #419: @17 = vertcat(@6, @1, @44) */
  rr=w17;
  *rr++ = w6;
  *rr++ = w44;
  /* #420: @16 = @17[:2] */
  for (rr=w16, ss=w17+0; ss!=w17+2; ss+=1) *rr++ = *ss;
  /* #421: (@50[2:8:3] = @16) */
  for (rr=w50+2, ss=w16; rr!=w50+8; rr+=3) *rr = *ss++;
  /* #422: @52 = @50' */
  casadi_trans(w50,casadi_s8, w52, casadi_s9, iw);
  /* #423: output[3][0] = @52 */
  casadi_copy(w52, 6, res[3]);
  return 0;
}

CASADI_SYMBOL_EXPORT int engine_impl_dae_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void engine_impl_dae_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void engine_impl_dae_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int engine_impl_dae_jac_x_xdot_u_z_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int engine_impl_dae_jac_x_xdot_u_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT const char* engine_impl_dae_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* engine_impl_dae_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* engine_impl_dae_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s10;
    case 1: return casadi_s10;
    case 2: return casadi_s11;
    case 3: return casadi_s11;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* engine_impl_dae_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s5;
    case 2: return casadi_s7;
    case 3: return casadi_s9;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int engine_impl_dae_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 8;
  if (sz_res) *sz_res = 6;
  if (sz_iw) *sz_iw = 5;
  if (sz_w) *sz_w = 89;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
