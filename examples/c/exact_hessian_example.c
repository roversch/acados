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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acados/sim/allocate_sim.h"
#include "acados/sim/casadi_wrapper.h"
#include "acados/sim/sim_erk_integrator.h"
#include "acados/sim/sim_common.h"
#include "acados/sim/sim_rk_common.h"
#include "acados/ocp_nlp/allocate_ocp_nlp.h"
#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados/ocp_nlp/ocp_nlp_eh_sqp.h"
#include "acados/utils/print.h"

int adj_hess_toy(const real_t **arg, real_t **res, int *iw, real_t *w, int mem);
int vde_toy(const real_t **arg, real_t **res, int *iw, real_t *w, int mem);

int main() {

    // Problem-specific data
    int_t N = 1, NX = 1, NU = 1;
    real_t W[] = {1.0, 0.0, 0.0, 1.0};
    real_t ref[] = {0.0, 0.0};
    real_t x0[] = {0.8};

    // Problem dimensions
    int_t nx[N+1], nu[N+1], nb[N+1], nc[N+1], ng[N+1];

    for (int_t k = 0; k <= N; k++) {
        nx[k] = NX;
        nu[k] = NU;
        nb[k] = 0;
        nc[k] = 0;
        ng[k] = 0;
    }
    nu[N] = 0;
    nb[0] = NX;

    ocp_nlp_in nlp;
    allocate_ocp_nlp_in(N, nx, nu, nb, nc, ng, 1, &nlp);
    int_t idxb0[] = {0};
    nlp.idxb[0] = idxb0;

    // Cost function
    for (int_t k = 0; k <= N; k++) {
        memcpy(((ocp_nlp_ls_cost *) nlp.cost)->W[k], W, (nx[k]+nu[k])*(nx[k]+nu[k])*sizeof(real_t));
        memcpy(((ocp_nlp_ls_cost *) nlp.cost)->y_ref[k], ref, (nx[k]+nu[k])*sizeof(real_t));
    }

    // Dynamics
    for (int_t k = 0; k < N; k++) {
        sim_in *sim = nlp.sim[k].in;
        sim->nx = NX;
        sim->nu = NU;
        sim->step = 0.3;
        sim->num_steps = 1;
        sim->forward_vde_wrapper = vde_fun;
        sim->sens_forw = true;
        sim->num_forw_sens = NX+NU;
        sim->vde = vde_toy;
        sim->adjoint_vde_wrapper = vde_hess_fun;
        sim->sens_adj = true;
        sim->sens_hess = true;
        sim->vde_adj = adj_hess_toy;
        sim->x[0] = 0;
        sim->u[0] = 1;
        sim_RK_opts *args = (sim_RK_opts *) malloc(sizeof(sim_RK_opts));
        sim_erk_create_arguments(args, 4);
        int_t work_size = sim_erk_calculate_workspace_size(sim, args);
        void *work = malloc(work_size);
        sim_erk_initialize(sim, args, &work);
        nlp.sim[k].fun = sim_erk;
        nlp.sim[k].args = args;
        nlp.sim[k].work = work;
        nlp.sim[k].fun(sim, nlp.sim[k].out, nlp.sim[k].args, nlp.sim[k].mem, nlp.sim[k].work);
        printf("out: %f\n", nlp.sim[k].out->xn[0]);
    }

    ocp_nlp_out output;
    allocate_ocp_nlp_out(&nlp, &output);

    ocp_nlp_eh_sqp_args nlp_args;
    ocp_nlp_args nlp_common_args;
    nlp_args.common = &nlp_common_args;
    nlp_args.common->maxIter = 10;
    snprintf(nlp_args.qp_solver_name, sizeof(nlp_args.qp_solver_name), "%s", "condensing_qpoases");

    ocp_nlp_eh_sqp_memory nlp_mem;
    ocp_nlp_memory nlp_mem_common;
    nlp_mem.common = &nlp_mem_common;
    ocp_nlp_eh_sqp_create_memory(&nlp, &nlp_args, &nlp_mem);

    int_t work_space_size = ocp_nlp_eh_sqp_calculate_workspace_size(&nlp, &nlp_args);
    void *nlp_work = malloc(work_space_size);

    // Initial guess
    for (int_t k = 0; k <= N; k++) {
        for (int_t j = 0; j < nx[k]; j++)
            nlp_mem.common->x[k][j] = 0.0;
        for (int_t j = 0; j < nu[k]; j++)
            nlp_mem.common->u[k][j] = 0.0;
    }

    nlp.lb[0] = x0;
    nlp.ub[0] = x0;
    int_t status = ocp_nlp_eh_sqp(&nlp, &output, &nlp_args, &nlp_mem, nlp_work);
    printf("\n\nstatus = %i\n\n", status);

    for (int_t k = 0; k <= N; k++) {
        char states_name[MAX_STR_LEN], controls_name[MAX_STR_LEN];
        snprintf(states_name, sizeof(states_name), "x%d.txt", k);
        print_matrix_name("stdout", states_name, output.x[k], 1, nx[k]);
        snprintf(controls_name, sizeof(controls_name), "u%d.txt", k);
        print_matrix_name("stdout", controls_name, output.u[k], 1, nu[k]);
    }
}
