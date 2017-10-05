/*
 *    This file is part of acados.
 *
 *    acados is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    acados is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with acados; if not, write to the Free Software Foundation,
 *    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_i_aux_ext_dep.h"

#include "acados/ocp_nlp/allocate_ocp_nlp.h"
#include "acados/ocp_nlp/ocp_nlp_eh_sqp.h"
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/sim/casadi_wrapper.h"
#include "acados/sim/sim_common.h"
#include "acados/sim/sim_erk_integrator.h"
#include "acados/sim/sim_lifted_irk_integrator.h"
#include "acados/utils/print.h"
#include "acados/utils/timing.h"
#include "acados/utils/types.h"
#include "examples/c/chain_model/chain_model.h"

#define NN 15
#define TT 3.0
#define Ns 2

enum sensitivities_scheme {
    EXACT_NEWTON,
    INEXACT_NEWTON,
    INIS,
    FROZEN_INEXACT_NEWTON,
    FROZEN_INIS
};

static void print_problem_info(enum sensitivities_scheme sensitivities_type,
                               const int_t num_free_masses, const int_t num_stages) {
    char scheme_name[MAX_STR_LEN];
    switch(sensitivities_type) {
        case EXACT_NEWTON:
            snprintf(scheme_name, sizeof(scheme_name), "EXACT_NEWTON");
            break;
        case INEXACT_NEWTON:
            snprintf(scheme_name, sizeof(scheme_name), "INEXACT_NEWTON");
            break;
        case INIS:
            snprintf(scheme_name, sizeof(scheme_name), "INIS");
            break;
        case FROZEN_INEXACT_NEWTON:
            snprintf(scheme_name, sizeof(scheme_name), "FROZEN_INEXACT_NEWTON");
            break;
        case FROZEN_INIS:
            snprintf(scheme_name, sizeof(scheme_name), "FROZEN_INIS");
            break;
        default:
            printf("Chose sensitivities type not available");
            exit(1);
    }
    printf("\n----- NUMBER OF FREE MASSES = %d, stages = %d (%s) -----\n",
           num_free_masses, num_stages, scheme_name);
}

static void select_model(const int_t num_free_masses, sim_in *sim) {
    switch (num_free_masses) {
        case 1:
            sim->vde = &vde_chain_nm2;
            sim->forward_vde_wrapper = &vde_fun;
            sim->jac = &jac_chain_nm2;
            sim->jacobian_wrapper = &jac_fun;
            sim->vde_adj = &vde_hess_chain_nm2;
            sim->adjoint_vde_wrapper = &vde_hess_fun;
            break;
        case 2:
            sim->vde = &vde_chain_nm3;
            sim->forward_vde_wrapper = &vde_fun;
            sim->jac = &jac_chain_nm3;
            sim->jacobian_wrapper = &jac_fun;
            sim->vde_adj = &vde_hess_chain_nm3;
            sim->adjoint_vde_wrapper = &vde_hess_fun;
            break;
        case 3:
            sim->vde = &vde_chain_nm4;
            sim->forward_vde_wrapper = &vde_fun;
            sim->jac = &jac_chain_nm4;
            sim->jacobian_wrapper = &jac_fun;
            sim->vde_adj = &vde_hess_chain_nm4;
            sim->adjoint_vde_wrapper = &vde_hess_fun;
            break;
        default:
            printf("Problem size not available");
            exit(1);
            break;
    }
}

void read_initial_state(const int_t nx, const int_t num_free_masses, real_t *x0) {
    FILE *initial_states_file;
    switch (num_free_masses) {
        case 1:
            initial_states_file = fopen(X0_NM2_FILE, "r");
            break;
        case 2:
            initial_states_file = fopen(X0_NM3_FILE, "r");
            break;
        case 3:
            initial_states_file = fopen(X0_NM4_FILE, "r");
            break;
        case 4:
            initial_states_file = fopen(X0_NM5_FILE, "r");
            break;
        case 5:
            initial_states_file = fopen(X0_NM6_FILE, "r");
            break;
        case 6:
            initial_states_file = fopen(X0_NM7_FILE, "r");
            break;
        case 7:
            initial_states_file = fopen(X0_NM8_FILE, "r");
            break;
        default:
            initial_states_file = fopen(X0_NM9_FILE, "r");
            break;
    }
    for (int_t i = 0; i < nx; i++)
        fscanf(initial_states_file, "%lf", &x0[i]);
    fclose(initial_states_file);
}

void read_final_state(const int_t nx, const int_t num_free_masses, real_t *xN) {
    FILE *final_state_file;
    switch (num_free_masses) {
        case 1:
            final_state_file = fopen(XN_NM2_FILE, "r");
            break;
        case 2:
            final_state_file = fopen(XN_NM3_FILE, "r");
            break;
        case 3:
            final_state_file = fopen(XN_NM4_FILE, "r");
            break;
        case 4:
            final_state_file = fopen(XN_NM5_FILE, "r");
            break;
        case 5:
            final_state_file = fopen(XN_NM6_FILE, "r");
            break;
        case 6:
            final_state_file = fopen(XN_NM7_FILE, "r");
            break;
        case 7:
            final_state_file = fopen(XN_NM8_FILE, "r");
            break;
        default:
            final_state_file = fopen(XN_NM9_FILE, "r");
            break;
    }
    for (int_t i = 0; i < nx; i++)
        fscanf(final_state_file, "%lf", &xN[i]);
    fclose(final_state_file);
}

int main() {
    // TODO(dimitris): fix for NMF > 1
    enum sensitivities_scheme scheme = EXACT_NEWTON;
    const int NMF = 2;
    const int d = 0;
    print_problem_info(scheme, NMF, d);

    // Dimensions
    int_t NX = 6 * NMF;
    int_t NU = 3;

    int_t nx[NN + 1] = {0};
    int_t nu[NN + 1] = {0};
    int_t nb[NN + 1] = {0};
    int_t nc[NN + 1] = {0};
    int_t ng[NN + 1] = {0};
    for (int_t i = 0; i < NN; i++) {
        nx[i] = NX;
        nu[i] = NU;
        nb[i] = NMF + NU;
    }
    nx[NN] = NX;
    nb[0] = NX + NU;
    nb[NN] = NX;

    // Problem data
    real_t wall_pos = -0.01;
    real_t UMAX = 10;
    real_t lb0[NX+NU], ub0[NX+NU];
    read_initial_state(NX, NMF, lb0);
    read_initial_state(NX, NMF, ub0);
    for (int_t i = NX; i < NX+NU; i++) {
        lb0[i] = -UMAX;
        ub0[i] = +UMAX;
    }
    real_t lb[NMF+NU], ub[NMF+NU];
    real_t xref[NX];
    read_final_state(NX, NMF, xref);
    real_t uref[3] = {0.0, 0.0, 0.0};
    real_t diag_cost_x[NX];
    for (int_t i = 0; i < NX; i++)
        diag_cost_x[i] = 1e-2;
    real_t diag_cost_u[3] = {1.0, 1.0, 1.0};
    real_t x_pos_inf[NX], x_neg_inf[NX];
    for (int_t i = 0; i < NX; i++) {
        x_pos_inf[i] = +1e4;
        x_neg_inf[i] = -1e4;
    }

    ocp_nlp_in *nlp;
    nlp = (ocp_nlp_in *) malloc(sizeof(ocp_nlp_in));
    allocate_ocp_nlp_in(NN, nx, nu, nb, nc, ng, d, nlp);

    // Least-squares cost
    ocp_nlp_ls_cost *ls_cost = (ocp_nlp_ls_cost *) nlp->cost;
    for (int_t i = 0; i < NN; i++) {
        for (int_t j = 0; j < NX; j++)
            ls_cost->W[i][j * (NX + NU + 1)] = diag_cost_x[j];
        for (int_t j = 0; j < NU; j++)
            ls_cost->W[i][(NX + j) * (NX + NU + 1)] = diag_cost_u[j];
        for (int_t j = 0; j < NX; j++)
            ls_cost->y_ref[i][j] = xref[j];
        for (int_t j = 0; j < NU; j++)
            ls_cost->y_ref[i][NX+j] = uref[j];
    }
    for (int_t j = 0; j < NX; j++)
        ((ocp_nlp_ls_cost *) nlp->cost)->W[NN][j * (NX + 1)] = diag_cost_x[j];

    // Simulation
    real_t Ts = TT / NN;
    sim_RK_opts rk_opts[NN];
    sim_lifted_irk_memory irk_mem[NN];
    nlp->freezeSens = false;
    if (scheme > 2)
        nlp->freezeSens = true;

    for (int_t jj = 0; jj < NN; jj++) {
        nlp->sim[jj].in->num_steps = Ns;
        nlp->sim[jj].in->step = Ts / nlp->sim[jj].in->num_steps;
        nlp->sim[jj].in->nx = NX;
        nlp->sim[jj].in->nu = NU;

        nlp->sim[jj].in->sens_forw = true;
        nlp->sim[jj].in->sens_adj = true;
        nlp->sim[jj].in->sens_hess = true;
        nlp->sim[jj].in->num_forw_sens = NX + NU;

        select_model(NMF, nlp->sim[jj].in);

        for (int_t i = 0; i < NX * (NX + NU); i++)
            nlp->sim[jj].in->S_forw[i] = 0.0;
        for (int_t i = 0; i < NX; i++)
            nlp->sim[jj].in->S_forw[i * (NX + 1)] = 1.0;
        for (int_t i = 0; i < NX + NU; i++)
            nlp->sim[jj].in->S_adj[i] = 0.0;
        for (int_t i = 0; i < d * NX; i++)
            nlp->sim[jj].in->grad_K[i] = 0.0;

        nlp->sim[jj].args = &rk_opts[jj];

        int_t workspace_size;
        if (d > 0) {
            nlp->sim[jj].fun = &sim_lifted_irk;
            nlp->sim[jj].mem = &irk_mem[jj];
            sim_irk_create_arguments(&rk_opts[jj], d, "Gauss");
            if (scheme == EXACT_NEWTON) {
                sim_irk_create_Newton_scheme(&rk_opts[jj], d, "Gauss", exact);
            } else if (scheme == INEXACT_NEWTON || scheme == FROZEN_INEXACT_NEWTON) {
                sim_irk_create_Newton_scheme(&rk_opts[jj], d, "Gauss", simplified_in);
            } else if (scheme == INIS || scheme == FROZEN_INIS) {
                sim_irk_create_Newton_scheme(&rk_opts[jj], d, "Gauss", simplified_inis);
            }
            sim_lifted_irk_create_memory(nlp->sim[jj].in, &rk_opts[jj], &irk_mem[jj]);
            workspace_size = sim_lifted_irk_calculate_workspace_size(nlp->sim[jj].in, &rk_opts[jj]);
        } else {
            nlp->sim[jj].fun = &sim_erk;
            nlp->sim[jj].mem = 0;
            sim_erk_create_arguments(&rk_opts[jj], 4);
            workspace_size = sim_erk_calculate_workspace_size(nlp->sim[jj].in, &rk_opts[jj]);
        }
        nlp->sim[jj].work = (void *) malloc(workspace_size);
    }

    // Box constraints
    int_t idxb_0[nb[0]], idxb_1[nb[1]], idxb_N[nb[NN]];
    for (int_t i = 0; i < nb[0]; i++)
        idxb_0[i] = i;
    for (int_t i = 0; i < NMF; i++)
        idxb_1[i] = 6*i + 1;
    for (int_t i = 0; i < NU; i++)
        idxb_1[NMF+i] = NX+i;
    for (int_t i = 0; i < nb[NN]; i++)
        idxb_N[i] = i;
    nlp->lb[0] = lb0;
    nlp->ub[0] = ub0;
    nlp->idxb[0] = idxb_0;
    for (int_t j = 0; j < NMF; j++) {
        lb[j] = wall_pos;  // wall position
        ub[j] = 1e4;
    }
    for (int_t j = 0; j < NU; j++) {
        lb[NMF+j] = -UMAX;  // umin
        ub[NMF+j] = +UMAX;  // umax
    }
    for (int_t i = 1; i < NN; i++) {
        nlp->lb[i] = lb;
        nlp->ub[i] = ub;
        nlp->idxb[i] = idxb_1;
    }
    nlp->lb[NN] = x_neg_inf;
    nlp->ub[NN] = x_pos_inf;
    nlp->idxb[NN] = idxb_N;

    ocp_nlp_out *nlp_out = (ocp_nlp_out *) malloc(sizeof(ocp_nlp_out));
    allocate_ocp_nlp_out(nlp, nlp_out);

    ocp_nlp_eh_sqp_args *nlp_args = (ocp_nlp_eh_sqp_args *) malloc(sizeof(ocp_nlp_eh_sqp_args));
    ocp_nlp_args *nlp_common_args = (ocp_nlp_args *) malloc(sizeof(ocp_nlp_args));
    nlp_args->common = nlp_common_args;
    nlp_args->common->maxIter = 100;
    snprintf(nlp_args->qp_solver_name, sizeof(nlp_args->qp_solver_name), "%s", "hpipm");

    ocp_nlp_eh_sqp_memory *nlp_mem = (ocp_nlp_eh_sqp_memory *) malloc(sizeof(ocp_nlp_eh_sqp_memory));
    ocp_nlp_memory *nlp_mem_common = (ocp_nlp_memory *) malloc(sizeof(ocp_nlp_memory));
    nlp_mem->common = nlp_mem_common;
    ocp_nlp_eh_sqp_create_memory(nlp, nlp_args, nlp_mem);

    int_t work_space_size = ocp_nlp_eh_sqp_calculate_workspace_size(nlp, nlp_args);
    void *nlp_work = (void *) malloc(work_space_size);

    // Initial guess
    for (int_t i = 0; i < NN; i++) {
        for (int_t j = 0; j < NX; j++)
            nlp_mem->common->x[i][j] = xref[j];
        for (int_t j = 0; j < NU; j++)
            nlp_mem->common->u[i][j] = 0.0;
    }
    for (int_t j = 0; j < NX; j++)
        nlp_mem->common->x[NN][j] = xref[j];

    int_t status = ocp_nlp_eh_sqp(nlp, nlp_out, nlp_args, nlp_mem, nlp_work);
    printf("\n\nstatus = %i\n\n", status);

    for (int_t k = 0; k <= NN; k++) {
        char states_name[MAX_STR_LEN], controls_name[MAX_STR_LEN];
        snprintf(states_name, sizeof(states_name), "x%d", k);
        print_matrix_name("stdout", states_name, nlp_out->x[k], 1, nx[k]);
        snprintf(controls_name, sizeof(controls_name), "u%d", k);
        print_matrix_name("stdout", controls_name, nlp_out->u[k], 1, nu[k]);
    }
}
