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

#include "acados/ocp_nlp/ocp_nlp_eh_sqp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados/sim/sim_common.h"
#include "acados/utils/math.h"
#include "acados/utils/print.h"
#include "acados/utils/timing.h"
#include "acados/utils/types.h"

int_t ocp_nlp_eh_sqp_calculate_workspace_size(const ocp_nlp_in *in, void *args_) {
    ocp_nlp_eh_sqp_args *args = (ocp_nlp_eh_sqp_args*) args_;

    int_t size;

    size = sizeof(ocp_nlp_eh_sqp_work);
    size += ocp_nlp_calculate_workspace_size(in, args->common);
    return size;
}

static void ocp_nlp_eh_sqp_cast_workspace(ocp_nlp_eh_sqp_work *work,
                                          ocp_nlp_eh_sqp_memory *mem) {
    char *ptr = (char *)work;

    ptr += sizeof(ocp_nlp_eh_sqp_work);
    work->common = (ocp_nlp_work *)ptr;
    ocp_nlp_cast_workspace(work->common, mem->common);
}


static void initialize_objective(
    const ocp_nlp_in *nlp_in,
    ocp_nlp_eh_sqp_memory *eh_sqp_mem,
    ocp_nlp_eh_sqp_work *work) {

    const int_t N = nlp_in->N;
    const int_t *nx = nlp_in->nx;
    const int_t *nu = nlp_in->nu;
    ocp_nlp_ls_cost *cost = (ocp_nlp_ls_cost*) nlp_in->cost;

    real_t **qp_Q = (real_t **) eh_sqp_mem->qp_solver->qp_in->Q;
    real_t **qp_S = (real_t **) eh_sqp_mem->qp_solver->qp_in->S;
    real_t **qp_R = (real_t **) eh_sqp_mem->qp_solver->qp_in->R;
    // TODO(rien): only for least squares cost with state and control reference atm
    for (int_t i = 0; i <= N; i++) {
        for (int_t j = 0; j < nx[i]; j++) {
            for (int_t k = 0; k < nx[i]; k++) {
                qp_Q[i][j * nx[i] + k] = cost->W[i][j * (nx[i] + nu[i]) + k];
            }
            for (int_t k = 0; k < nu[i]; k++) {
                qp_S[i][j * nu[i] + k] =
                    cost->W[i][j * (nx[i] + nu[i]) + nx[i] + k];
            }
        }
        for (int_t j = 0; j < nu[i]; j++) {
            for (int_t k = 0; k < nu[i]; k++) {
                qp_R[i][j * nu[i] + k] =
                    cost->W[i][(nx[i] + j) * (nx[i] + nu[i]) + nx[i] + k];
            }
        }
    }
}


static void initialize_trajectories(
    const ocp_nlp_in *nlp_in,
    ocp_nlp_eh_sqp_memory *eh_sqp_mem,
    ocp_nlp_eh_sqp_work *work) {

    const int_t N = nlp_in->N;
    const int_t *nx = nlp_in->nx;
    const int_t *nu = nlp_in->nu;
    real_t *w = work->common->w;

    int_t w_idx = 0;
    for (int_t i = 0; i <= N; i++) {
        for (int_t j = 0; j < nx[i]; j++) {
            w[w_idx + j] = eh_sqp_mem->common->x[i][j];
        }
        for (int_t j = 0; j < nu[i]; j++) {
            w[w_idx + nx[i] + j] = eh_sqp_mem->common->u[i][j];
        }
        w_idx += nx[i] + nu[i];
    }
}


static void multiple_shooting(const ocp_nlp_in *nlp, ocp_nlp_eh_sqp_memory *mem, real_t *w) {

    const int_t N = nlp->N;
    const int_t *nx = nlp->nx;
    const int_t *nu = nlp->nu;
    sim_solver *sim = nlp->sim;
    ocp_nlp_ls_cost *cost = (ocp_nlp_ls_cost *) nlp->cost;
    real_t **y_ref = cost->y_ref;

    real_t **qp_A = (real_t **) mem->qp_solver->qp_in->A;
    real_t **qp_B = (real_t **) mem->qp_solver->qp_in->B;
    real_t **qp_b = (real_t **) mem->qp_solver->qp_in->b;
    real_t **qp_Q = (real_t **) mem->qp_solver->qp_in->Q;
    real_t **qp_S = (real_t **) mem->qp_solver->qp_in->S;
    real_t **qp_R = (real_t **) mem->qp_solver->qp_in->R;
    real_t **qp_q = (real_t **) mem->qp_solver->qp_in->q;
    real_t **qp_r = (real_t **) mem->qp_solver->qp_in->r;
    real_t **qp_lb = (real_t **) mem->qp_solver->qp_in->lb;
    real_t **qp_ub = (real_t **) mem->qp_solver->qp_in->ub;

    int_t w_idx = 0;

    for (int_t i = 0; i < N; i++) {
        // Pass state and control to integrator
        for (int_t j = 0; j < nx[i]; j++) sim[i].in->x[j] = w[w_idx+j];
        for (int_t j = 0; j < nu[i]; j++) sim[i].in->u[j] = w[w_idx+nx[i]+j];
        sim[i].fun(sim[i].in, sim[i].out, sim[i].args, sim[i].mem, sim[i].work);

        // TODO(rien): transition functions for changing dimensions not yet implemented!
        for (int_t j = 0; j < nx[i]; j++) {
            qp_b[i][j] = sim[i].out->xn[j] - w[w_idx+nx[i]+nu[i]+j];
            for (int_t k = 0; k < nx[i]; k++)
                qp_A[i][j*nx[i]+k] = sim[i].out->S_forw[j*nx[i]+k];
        }
        for (int_t j = 0; j < nu[i]; j++)
            for (int_t k = 0; k < nx[i]; k++)
                qp_B[i][j*nx[i]+k] = sim[i].out->S_forw[(nx[i]+j)*nx[i]+k];

        // Update bounds:
        for (int_t j = 0; j < nlp->nb[i]; j++) {
            qp_lb[i][j] = nlp->lb[i][j] - w[w_idx+nlp->idxb[i][j]];
            qp_ub[i][j] = nlp->ub[i][j] - w[w_idx+nlp->idxb[i][j]];
        }

        // Update gradients
        // TODO(rien): only for diagonal Q, R matrices atm
        sim_RK_opts *opts = (sim_RK_opts*) sim[i].args;
        int_t hess_index = 0;
        for (int_t j = 0; j < nx[i]; j++) {
            for (int_t k = 0; k < nx[i]; k++)
                qp_Q[i][j*nx[i] + k] = cost->W[i][j*(nx[i]+nu[i]) + k];
            for (int_t k = 0; k < nu[i]; k++)
                qp_S[i][j] = cost->W[i][j*(nx[i]+nu[i]) + nx[i] + k];
        }
        for (int_t j = 0; j < nu[i]; j++) {
            for (int_t k = 0; k < nu[i]; k++) {
                qp_R[i][j*nu[i] + k] = cost->W[i][(nx[i] + j)*(nx[i]+nu[i]) + nx[i] + k];
            }
        }

        for (int_t j = 0; j < nx[i]; j++) {
            for (int_t k = j; k < nx[i]; k++) {
                qp_Q[i][j*nx[i] + k] += sim[i].out->S_hess[hess_index];
                qp_Q[i][k*nx[i] + j] = qp_Q[i][j*nx[i] + k];
                hess_index++;
            }
            for (int_t k = 0; k < nu[i]; k++) {
                qp_S[i][j*nu[i] + k] += sim[i].out->S_hess[hess_index];
                hess_index++;
            }
        }
        for (int_t j = 0; j < nx[i]; j++) {
            qp_q[i][j] = 0;
            for (int_t k = 0; k < nx[i]+nu[i]; k++)
                qp_q[i][j] += cost->W[i][j+k*(nx[i]+nu[i])] * (w[w_idx+k]-y_ref[i][k]);
            // adjoint-based gradient correction:
            if (opts->scheme.type != exact)
                qp_q[i][j] += sim[i].out->grad[j];
        }
        for (int_t j = 0; j < nu[i]; j++) {
            for (int_t k = j; k < nu[i]; k++) {
                qp_R[i][j*nu[i] + k] += sim[i].out->S_hess[hess_index];
                qp_R[i][k*nu[i] + j] = qp_R[i][j*nu[i] + k];
                hess_index++;
            }
        }
        for (int_t j = 0; j < nu[i]; j++) {
            qp_r[i][j] = 0;
            for (int_t k = 0; k < nx[i]+nu[i]; k++)
                qp_r[i][j] += cost->W[i][nx[i]+j+k*(nx[i]+nu[i])] * (w[w_idx+k]-y_ref[i][k]);
            // adjoint-based gradient correction:
            if (opts->scheme.type != exact)
                qp_r[i][j] += sim[i].out->grad[nx[i]+j];
        }
        w_idx += nx[i]+nu[i];
    }

    for (int_t j = 0; j < nlp->nb[N]; j++) {
        qp_lb[N][j] = nlp->lb[N][j] - w[w_idx+nlp->idxb[N][j]];
        qp_ub[N][j] = nlp->ub[N][j] - w[w_idx+nlp->idxb[N][j]];
    }

    for (int_t j = 0; j < nx[N]; j++)
        for (int_t k = 0; k < nx[N]; k++)
            qp_Q[N][j*nx[N] + k] = cost->W[N][j*(nx[N]+nu[N]) + k];

    for (int_t j = 0; j < nx[N]; j++)
        qp_q[N][j] = cost->W[N][j*(nx[N]+nu[N]+1)]*(w[w_idx+j]-y_ref[N][j]);
}


static void update_variables(const ocp_nlp_in *nlp, ocp_nlp_eh_sqp_memory *mem, real_t *w,
                             real_t *pi) {
    const int_t N = nlp->N;
    const int_t *nx = nlp->nx;
    const int_t *nu = nlp->nu;
    sim_solver *sim = nlp->sim;

    int_t pi_idx = 0;
    for (int_t i = 0; i < N; i++)
        for (int_t j = 0; j < nx[i+1]; j++) {
            pi[pi_idx+j] = mem->qp_solver->qp_out->pi[i][j];
            pi_idx += nx[i+1];
        }

    pi_idx = 0;
    for (int_t i = 0; i < N; i++)
        for (int_t j = 0; j < nx[i+1]; j++) {
            sim[i].in->S_adj[j] = pi[pi_idx+j];
            pi_idx += nx[i+1];
        }

    int_t w_idx = 0;
    for (int_t i = 0; i <= N; i++) {
        for (int_t j = 0; j < nx[i]; j++)
            w[w_idx+j] += mem->qp_solver->qp_out->x[i][j];
        for (int_t j = 0; j < nu[i]; j++)
            w[w_idx+nx[i]+j] += mem->qp_solver->qp_out->u[i][j];
        w_idx += nx[i]+nu[i];
    }
}


static void store_trajectories(const ocp_nlp_in *nlp, ocp_nlp_memory *memory, ocp_nlp_out *out,
    real_t *w) {

    const int_t N = nlp->N;
    const int_t *nx = nlp->nx;
    const int_t *nu = nlp->nu;

    int_t w_idx = 0;
    for (int_t i = 0; i <= N; i++) {
        for (int_t j = 0; j < nx[i]; j++) {
            memory->x[i][j] = w[w_idx+j];
            out->x[i][j] = w[w_idx+j];
        }
        for (int_t j = 0; j < nu[i]; j++) {
            memory->u[i][j] = w[w_idx+nx[i]+j];
            out->u[i][j] = w[w_idx+nx[i]+j];
        }
        w_idx += nx[i] + nu[i];
    }
}

static void QSR_to_matrix(int_t nx, int_t nu, const real_t *Q, const real_t *S, const real_t *R,
                          real_t *matrix) {
    for (int_t i = 0; i < nx; i++)
        for (int_t j = 0; j < nx; j++)
            matrix[i*(nx+nu) + j] = Q[i*nx + j];
    for (int_t i = 0; i < nx; i++)
        for (int_t j = 0; j < nu; j++) {
            matrix[i*(nx+nu) + nx + j] = S[i*nu + j];
            matrix[(nx+j)*(nx+nu) + i] = S[i*nu + j];
        }
    for (int_t i = 0; i < nu; i++)
        for (int_t j = 0; j < nu; j++)
            matrix[(nx+i)*(nu+nx) + nx + j] = R[i*nu+j];
}

static void matrix_to_QSR(int_t nx, int_t nu, real_t *Q, real_t *S, real_t *R,
                          const real_t *matrix) {
    for (int_t i = 0; i < nx; i++)
        for (int_t j = 0; j < nx; j++)
            Q[i*nx + j] = matrix[i*(nx+nu) + j];
    for (int_t i = 0; i < nx; i++)
        for (int_t j = 0; j < nu; j++)
            S[i*nu + j] = matrix[i*(nx+nu) + nx + j];
    for (int_t i = 0; i < nu; i++)
        for (int_t j = 0; j < nu; j++)
            R[i*nu+j] = matrix[(nx+i)*(nx+nu) + nx + j];
}

static void hessian_regularization(ocp_nlp_eh_sqp_memory *mem) {
    ocp_qp_in *qp = mem->qp_solver->qp_in;
    for (int_t i = 0; i <= qp->N; i++) {
        int_t n = qp->nx[i] + qp->nu[i];
        real_t *matrix = calloc(n*n, sizeof(real_t));
        QSR_to_matrix(qp->nx[i], qp->nu[i], qp->Q[i], qp->S[i], qp->R[i], matrix);
        regularize(n, matrix);
        matrix_to_QSR(qp->nx[i], qp->nu[i], (real_t *) qp->Q[i], (real_t *) qp->S[i],
                      (real_t *) qp->R[i], matrix);
    }
}


// Simple fixed-step Gauss-Newton based SQP routine
int_t ocp_nlp_eh_sqp(const ocp_nlp_in *nlp_in, ocp_nlp_out *nlp_out, void *nlp_args_,
    void *nlp_mem_, void *nlp_work_) {

    real_t *pi = calloc(nlp_in->N*nlp_in->nx[0], sizeof(real_t));

    ocp_nlp_eh_sqp_memory *eh_sqp_mem = (ocp_nlp_eh_sqp_memory *) nlp_mem_;
    ocp_nlp_eh_sqp_work *work = (ocp_nlp_eh_sqp_work*) nlp_work_;
    ocp_nlp_eh_sqp_cast_workspace(work, eh_sqp_mem);

    initialize_objective(nlp_in, eh_sqp_mem, work);
    initialize_trajectories(nlp_in, eh_sqp_mem, work);

    // TODO(roversch): Do we need this here?
    int_t **qp_idxb = (int_t **) eh_sqp_mem->qp_solver->qp_in->idxb;
    for (int_t i = 0; i <= nlp_in->N; i++) {
        for (int_t j = 0; j < nlp_in->nb[i]; j++) {
            qp_idxb[i][j] = nlp_in->idxb[i][j];
        }
    }

    int_t max_sqp_iterations = ((ocp_nlp_eh_sqp_args *) nlp_args_)->common->maxIter;

    acados_timer timer;
    real_t total_time = 0;
    acados_tic(&timer);
    for (int_t sqp_iter = 0; sqp_iter < max_sqp_iterations; sqp_iter++) {

        multiple_shooting(nlp_in, eh_sqp_mem, work->common->w);

        hessian_regularization(eh_sqp_mem);

        int_t qp_status = eh_sqp_mem->qp_solver->fun(
            eh_sqp_mem->qp_solver->qp_in,
            eh_sqp_mem->qp_solver->qp_out,
            eh_sqp_mem->qp_solver->args,
            eh_sqp_mem->qp_solver->mem,
            eh_sqp_mem->qp_solver->work);

        if (qp_status != 0) {
            printf("QP solver returned error status %d\n", qp_status);
            return -1;
        }

        real_t inf_norm = 0;
        for (int_t i = 0; i <= nlp_in->N; i++) {
            for (int_t j = 0; j < nlp_in->nx[i]; j++)
                if (fabs(eh_sqp_mem->qp_solver->qp_out->x[i][j]) > inf_norm)
                    inf_norm = fabs(eh_sqp_mem->qp_solver->qp_out->x[i][j]);
            for (int_t j = 0; j < nlp_in->nu[i]; j++)
                if (fabs(eh_sqp_mem->qp_solver->qp_out->u[i][j]) > inf_norm)
                    inf_norm = fabs(eh_sqp_mem->qp_solver->qp_out->u[i][j]);
        }

        update_variables(nlp_in, eh_sqp_mem, work->common->w, pi);

        for (int_t i = 0; i < nlp_in->N; i++) {
            sim_RK_opts *opts = nlp_in->sim[i].args;
            nlp_in->sim[i].in->sens_adj = (nlp_in->sim[i].in->sens_hess)
                                                        || (opts->scheme.type != exact);
            if (nlp_in->freezeSens)  // freeze inexact sensitivities after first SQP iteration !!
                opts->scheme.freeze = true;
        }
    }

    total_time += acados_toc(&timer);
    store_trajectories(nlp_in, eh_sqp_mem->common, nlp_out, work->common->w);
    return 0;
}

void ocp_nlp_eh_sqp_create_memory(const ocp_nlp_in *in, void *args_, void *memory_) {

    ocp_nlp_eh_sqp_args *args = (ocp_nlp_eh_sqp_args *)args_;
    ocp_nlp_eh_sqp_memory *mem = (ocp_nlp_eh_sqp_memory *)memory_;

    ocp_qp_in *dummy_qp = create_ocp_qp_in(in->N, in->nx, in->nu, in->nb, in->nc);
    int_t **idxb = (int_t **) dummy_qp->idxb;
    for (int_t i = 0; i < in->N; i++)
        for (int_t j = 0; j < in->nb[i]; j++)
            idxb[i][j] = in->idxb[i][j];
    mem->qp_solver = create_ocp_qp_solver(dummy_qp, args->qp_solver_name, NULL);

    ocp_nlp_create_memory(in, mem->common);
}

void ocp_nlp_eh_sqp_free_memory(void *mem_) {
    ocp_nlp_eh_sqp_memory *mem = (ocp_nlp_eh_sqp_memory *)mem_;

    int_t N = mem->qp_solver->qp_in->N;
    ocp_nlp_free_memory(N, mem->common);

    mem->qp_solver->destroy(mem->qp_solver->mem, mem->qp_solver->work);

    free(mem->qp_solver->qp_in);
    free(mem->qp_solver->qp_out);
    free(mem->qp_solver->args);
    free(mem->qp_solver);
    // TODO(dimitris): where do we free the integrators?
}
