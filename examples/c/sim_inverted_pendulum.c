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

/*  This example shows how acados can be used to simulate index-1 DAEs
        using IRK integrators, either the standard sim_irk_integrator or
        the GNSF (generalized nonlinear static feedback) exploitingintegrator.
    Author: Jonathan Frey                                                   */

// external
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// acados
#include "acados/sim/sim_common.h"
#include "acados/sim/sim_gnsf.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/math.h"

#include "acados_c/external_function_interface.h"
#include "interfaces/acados_c/sim_interface.h"

// crane dae model
#include "examples/c/inverted_pendulum_model/inverted_pendulum_model.h"



int main()
{

	/************************************************
	* initialization
	************************************************/

    const int nx    = 6;
    const int nu    = 1;
    const int nz    = 5;

    const int nx1   = 5;  // gnsf split
    const int nz1   = 5;
    const int n_out = 3;
    const int ny    = 8;
    const int nuhat = 1;

	int nsim = 1;

    // generate x0, u_sim
    double x0[nx];
    double u_sim[nu];

    x0[0] =  1.0000;  // xpos
    x0[1] = -5.0000;  // ypos
    x0[2] =  0.1000;  // vx
    x0[3] = -0.5000;  // vy
    x0[4] =  0.1000;  // valpha
    x0[5] =  1.0000;  // alpha


    u_sim[0] = 1;

    int NF = nx + nu;  // columns of forward seed

    int nsim0 = 1;  // nsim;

    double T = 0.01;

	double *x_sim = malloc(sizeof(double)*nx*(nsim+1));

    for (int ii = 0; ii < nx; ii++)
        x_sim[ii] = x0[ii];

/************************************************
* external functions
************************************************/

    // impl_ode_fun
    external_function_casadi impl_ode_fun;
    impl_ode_fun.casadi_fun = &inv_pendulum_impl_ode_fun;
    impl_ode_fun.casadi_work = &inv_pendulum_impl_ode_fun_work;
    impl_ode_fun.casadi_sparsity_in = &inv_pendulum_impl_ode_fun_sparsity_in;
    impl_ode_fun.casadi_sparsity_out = &inv_pendulum_impl_ode_fun_sparsity_out;
    impl_ode_fun.casadi_n_in = &inv_pendulum_impl_ode_fun_n_in;
    impl_ode_fun.casadi_n_out = &inv_pendulum_impl_ode_fun_n_out;
    external_function_casadi_create(&impl_ode_fun);

    // impl_ode_fun_jac_x_xdot
    external_function_casadi impl_ode_fun_jac_x_xdot;
    impl_ode_fun_jac_x_xdot.casadi_fun = &inv_pendulum_impl_ode_fun_jac_x_xdot;
    impl_ode_fun_jac_x_xdot.casadi_work = &inv_pendulum_impl_ode_fun_jac_x_xdot_work;
    impl_ode_fun_jac_x_xdot.casadi_sparsity_in = &inv_pendulum_impl_ode_fun_jac_x_xdot_sparsity_in;
    impl_ode_fun_jac_x_xdot.casadi_sparsity_out = &inv_pendulum_impl_ode_fun_jac_x_xdot_sparsity_out;
    impl_ode_fun_jac_x_xdot.casadi_n_in = &inv_pendulum_impl_ode_fun_jac_x_xdot_n_in;
    impl_ode_fun_jac_x_xdot.casadi_n_out = &inv_pendulum_impl_ode_fun_jac_x_xdot_n_out;
    external_function_casadi_create(&impl_ode_fun_jac_x_xdot);

    // impl_ode_jac_x_xdot_u
    external_function_casadi impl_ode_jac_x_xdot_u;
    impl_ode_jac_x_xdot_u.casadi_fun = &inv_pendulum_impl_ode_jac_x_xdot_u;
    impl_ode_jac_x_xdot_u.casadi_work = &inv_pendulum_impl_ode_jac_x_xdot_u_work;
    impl_ode_jac_x_xdot_u.casadi_sparsity_in = &inv_pendulum_impl_ode_jac_x_xdot_u_sparsity_in;
    impl_ode_jac_x_xdot_u.casadi_sparsity_out = &inv_pendulum_impl_ode_jac_x_xdot_u_sparsity_out;
    impl_ode_jac_x_xdot_u.casadi_n_in = &inv_pendulum_impl_ode_jac_x_xdot_u_n_in;
    impl_ode_jac_x_xdot_u.casadi_n_out = &inv_pendulum_impl_ode_jac_x_xdot_u_n_out;
    external_function_casadi_create(&impl_ode_jac_x_xdot_u);

    // impl_ode_jac_x_xdot_u
    external_function_casadi impl_ode_fun_jac_x_xdot_u;
    impl_ode_fun_jac_x_xdot_u.casadi_fun = &inv_pendulum_impl_ode_fun_jac_x_xdot_u;
    impl_ode_fun_jac_x_xdot_u.casadi_work = &inv_pendulum_impl_ode_fun_jac_x_xdot_u_work;
    impl_ode_fun_jac_x_xdot_u.casadi_sparsity_in =
                            &inv_pendulum_impl_ode_fun_jac_x_xdot_u_sparsity_in;
    impl_ode_fun_jac_x_xdot_u.casadi_sparsity_out =
                            &inv_pendulum_impl_ode_fun_jac_x_xdot_u_sparsity_out;
    impl_ode_fun_jac_x_xdot_u.casadi_n_in = &inv_pendulum_impl_ode_fun_jac_x_xdot_u_n_in;
    impl_ode_fun_jac_x_xdot_u.casadi_n_out = &inv_pendulum_impl_ode_fun_jac_x_xdot_u_n_out;
    external_function_casadi_create(&impl_ode_fun_jac_x_xdot_u);

    // impl_ode_hess
    external_function_casadi impl_ode_hess;
    impl_ode_hess.casadi_fun = &inv_pendulum_impl_ode_hess;
    impl_ode_hess.casadi_work = &inv_pendulum_impl_ode_hess_work;
    impl_ode_hess.casadi_sparsity_in = &inv_pendulum_impl_ode_hess_sparsity_in;
    impl_ode_hess.casadi_sparsity_out = &inv_pendulum_impl_ode_hess_sparsity_out;
    impl_ode_hess.casadi_n_in = &inv_pendulum_impl_ode_hess_n_in;
    impl_ode_hess.casadi_n_out = &inv_pendulum_impl_ode_hess_n_out;
    external_function_casadi_create(&impl_ode_hess);

    /************************************************
    * external functions (Generalized Nonlinear Static Feedback (GNSF) model)
    ************************************************/
    // phi_fun
    external_function_casadi phi_fun;
    phi_fun.casadi_fun            = &inv_pendulum_phi_fun;
    phi_fun.casadi_work           = &inv_pendulum_phi_fun_work;
    phi_fun.casadi_sparsity_in    = &inv_pendulum_phi_fun_sparsity_in;
    phi_fun.casadi_sparsity_out   = &inv_pendulum_phi_fun_sparsity_out;
    phi_fun.casadi_n_in           = &inv_pendulum_phi_fun_n_in;
    phi_fun.casadi_n_out          = &inv_pendulum_phi_fun_n_out;
    external_function_casadi_create(&phi_fun);

    // phi_fun_jac_y
    external_function_casadi phi_fun_jac_y;
    phi_fun_jac_y.casadi_fun            = &inv_pendulum_phi_fun_jac_y;
    phi_fun_jac_y.casadi_work           = &inv_pendulum_phi_fun_jac_y_work;
    phi_fun_jac_y.casadi_sparsity_in    = &inv_pendulum_phi_fun_jac_y_sparsity_in;
    phi_fun_jac_y.casadi_sparsity_out   = &inv_pendulum_phi_fun_jac_y_sparsity_out;
    phi_fun_jac_y.casadi_n_in           = &inv_pendulum_phi_fun_jac_y_n_in;
    phi_fun_jac_y.casadi_n_out          = &inv_pendulum_phi_fun_jac_y_n_out;
    external_function_casadi_create(&phi_fun_jac_y);

    // phi_jac_y_uhat
    external_function_casadi phi_jac_y_uhat;
    phi_jac_y_uhat.casadi_fun                = &inv_pendulum_phi_jac_y_uhat;
    phi_jac_y_uhat.casadi_work               = &inv_pendulum_phi_jac_y_uhat_work;
    phi_jac_y_uhat.casadi_sparsity_in        = &inv_pendulum_phi_jac_y_uhat_sparsity_in;
    phi_jac_y_uhat.casadi_sparsity_out       = &inv_pendulum_phi_jac_y_uhat_sparsity_out;
    phi_jac_y_uhat.casadi_n_in               = &inv_pendulum_phi_jac_y_uhat_n_in;
    phi_jac_y_uhat.casadi_n_out              = &inv_pendulum_phi_jac_y_uhat_n_out;
    external_function_casadi_create(&phi_jac_y_uhat);

    // f_lo_fun_jac_x1k1uz
    external_function_casadi f_lo_fun_jac_x1k1uz;
    f_lo_fun_jac_x1k1uz.casadi_fun            = &inv_pendulum_f_lo_fun_jac_x1k1uz;
    f_lo_fun_jac_x1k1uz.casadi_work           = &inv_pendulum_f_lo_fun_jac_x1k1uz_work;
    f_lo_fun_jac_x1k1uz.casadi_sparsity_in    = &inv_pendulum_f_lo_fun_jac_x1k1uz_sparsity_in;
    f_lo_fun_jac_x1k1uz.casadi_sparsity_out   = &inv_pendulum_f_lo_fun_jac_x1k1uz_sparsity_out;
    f_lo_fun_jac_x1k1uz.casadi_n_in           = &inv_pendulum_f_lo_fun_jac_x1k1uz_n_in;
    f_lo_fun_jac_x1k1uz.casadi_n_out          = &inv_pendulum_f_lo_fun_jac_x1k1uz_n_out;
    external_function_casadi_create(&f_lo_fun_jac_x1k1uz);

    // get_matrices_fun
    external_function_casadi get_matrices_fun;
    get_matrices_fun.casadi_fun            = &inv_pendulum_get_matrices_fun;
    get_matrices_fun.casadi_work           = &inv_pendulum_get_matrices_fun_work;
    get_matrices_fun.casadi_sparsity_in    = &inv_pendulum_get_matrices_fun_sparsity_in;
    get_matrices_fun.casadi_sparsity_out   = &inv_pendulum_get_matrices_fun_sparsity_out;
    get_matrices_fun.casadi_n_in           = &inv_pendulum_get_matrices_fun_n_in;
    get_matrices_fun.casadi_n_out          = &inv_pendulum_get_matrices_fun_n_out;
    external_function_casadi_create(&get_matrices_fun);


/* nss: number of sim solver:
        2: IRK
        3: GNSF
                                */
	for (int nss = 2; nss < 4; nss++)
	{
		/************************************************
		* sim plan & config
		************************************************/

    /* choose plan */
		sim_solver_plan plan;
		switch (nss)
		{
			case 2:
				plan.sim_solver = IRK;
				break;

			case 3:
				plan.sim_solver = GNSF;
				break;

			default :
				printf("\nsolver not supported!\n");
				exit(1);
		}

		// create correct config based on plan
		sim_solver_config *config = sim_config_create(plan);

    /* sim dims */

		void *dims = sim_dims_create(config);
		config->set_nx(dims, nx);
		config->set_nu(dims, nu);
        config->set_nz(dims, nz);

        // GNSF -- set additional dimensions
        sim_gnsf_dims *gnsf_dim;
        if (plan.sim_solver == GNSF)
        {
            gnsf_dim = (sim_gnsf_dims *) dims;
            gnsf_dim->nx1   = nx1;
            gnsf_dim->nz1   = nz1;
            gnsf_dim->ny    = ny;
            gnsf_dim->nuhat = nuhat;
            gnsf_dim->n_out = n_out;
        }

    /* sim options */

        void *opts_ = sim_opts_create(config, dims);
        sim_rk_opts *opts = (sim_rk_opts *) opts_;
        config->opts_initialize_default(config, dims, opts);

        opts->jac_reuse = true;        // jacobian reuse
        opts->newton_iter = 3;          // number of newton iterations per integration step

        opts->ns                = 3;    // number of stages in rk integrator
        opts->num_steps         = 3;    // number of steps
        opts->sens_forw         = true;
        opts->sens_adj          = true;
        opts->output_z          = true;
        opts->sens_algebraic    = true;
        opts->sens_hess         = true;

    /* sim in / out */

		sim_in *in   = sim_in_create(config, dims);
		sim_out *out = sim_out_create(config, dims);

		in->T = T;

    /* set model */
        switch (plan.sim_solver)
        {
            case IRK:  // IRK
            {
                sim_set_model(config, in, "impl_ode_fun", &impl_ode_fun);
                sim_set_model(config, in, "impl_ode_fun_jac_x_xdot",
                        &impl_ode_fun_jac_x_xdot);
                sim_set_model(config, in, "impl_ode_jac_x_xdot_u", &impl_ode_jac_x_xdot_u);
                sim_set_model(config, in, "impl_ode_hess", &impl_ode_hess);
                break;
            }
            case GNSF:  // GNSF
            {
                // set model funtions
                sim_set_model(config, in, "phi_fun", &phi_fun);
                sim_set_model(config, in, "phi_fun_jac_y", &phi_fun_jac_y);
                sim_set_model(config, in, "phi_jac_y_uhat", &phi_jac_y_uhat);
                sim_set_model(config, in, "f_lo_jac_x1_x1dot_u_z", &f_lo_fun_jac_x1k1uz);

                // import model matrices
                external_function_generic *get_model_matrices =
                        (external_function_generic *) &get_matrices_fun;
                gnsf_model *model = (gnsf_model *) in->model;
                sim_gnsf_import_matrices(gnsf_dim, model, get_model_matrices);
                break;
            }
            // case NEW_LIFTED_IRK:  // new_lifted_irk
            // {
            //     sim_set_model(config, in, "impl_ode_fun", &impl_ode_fun);
            //     sim_set_model(config, in, "impl_ode_fun_jac_x_xdot_u",
            //              &impl_ode_fun_jac_x_xdot_u);
            //     break;
            // }
            default :
            {
                printf("\nnot enough sim solvers implemented!\n");
                exit(1);
            }
        }

    /* seeds */
        for (int ii = 0; ii < nx * NF; ii++)
            in->S_forw[ii] = 0.0;
        for (int ii = 0; ii < nx; ii++)
            in->S_forw[ii * (nx + 1)] = 1.0;

        // seeds adj
        for (int ii = 0; ii < nx; ii++)
            in->S_adj[ii] = 1.0;
        for (int ii = nx; ii < nx + nu; ii++)
            in->S_adj[ii] = 0.0;

    /* sim solver  */
        sim_solver *sim_solver = sim_create(config, dims, opts);
        int acados_return;

        if (plan.sim_solver == GNSF){  // for gnsf: perform precomputation
            gnsf_model *model = (gnsf_model *) in->model;
            sim_gnsf_precompute(config, gnsf_dim, model, opts,
                        sim_solver->mem, sim_solver->work, in->T);
        }

    /* print solver info */
        switch (plan.sim_solver)
        {
            case IRK:  // IRK
            {
				printf("\n\nsim solver: IRK");
				break;
            }
            case GNSF:  // GNSF
            {
				printf("\n\nsim solver: GNSF");
				break;
            }

			default :
				printf("\nsim solver not supported!\n");
				exit(1);

		}
        printf("\nns = %d \t num_steps = %d \t newton_iter = %d \t jac_reuse = %d \n",
                         opts->ns, opts->num_steps, opts->newton_iter, opts->jac_reuse);

		acados_timer timer;
		acados_tic(&timer);

    /* simulate */

		// int nsim0 = nsim;

		double cpu_time = 0.0;
		double la_time = 0.0;
		double ad_time = 0.0;

        for (int ii = 0; ii < nsim0; ii++)
		{
            // x
            for (int jj = 0; jj < nx; jj++)
                in->x[jj] = x_sim[ii*nx+jj];

            // u
            for (int jj = 0; jj < nu; jj++)
                in->u[jj] = u_sim[ii*nu+jj];

			// execute simulation step with current input and state
			acados_return = sim_solve(sim_solver, in, out);
			if (acados_return != 0)
			{
				printf("error in sim solver\n");
				return ACADOS_FAILURE;
			}

			cpu_time += out->info->CPUtime;
			la_time += out->info->LAtime;
			ad_time += out->info->ADtime;

			// d_print_mat(1, nx, out->xn, 1);
			// d_print_mat(1, nx, x_ref+ii*nx, 1);

			// extract state at next time step
			for (int jj = 0; jj < nx; jj++)
				x_sim[(ii+1)*nx+jj] = out->xn[jj];

		}
		double total_cpu_time = acados_toc(&timer);

		/************************************************
    * printing
		************************************************/
		printf("\nxn: \n");
		d_print_exp_mat(1, nx, &x_sim[nsim0*nx], 1);

		double *S_forw_out = NULL;
		if(opts->sens_forw){
			S_forw_out = out->S_forw;
			// printf("S_forw_out: \n");
			// d_print_exp_mat(nx, NF, S_forw_out, nx);
		}

		if(opts->sens_adj || opts->sens_hess){
			double *S_adj_out = out->S_adj;
			printf("S_adj_out: \n");
			d_print_exp_mat(1, nx+nu, S_adj_out, 1);
		}

		if(opts->sens_forw){		// debug adjoints
			struct blasfeo_dmat S_forw_result;
			struct blasfeo_dvec adjoint_seed;
			struct blasfeo_dvec forw_times_seed;

			blasfeo_allocate_dmat(nx, nx+nu, &S_forw_result);
			blasfeo_allocate_dvec(nx, &adjoint_seed);
			blasfeo_allocate_dvec(nx+nu, &forw_times_seed);

			blasfeo_pack_dmat(nx, nx+nu, S_forw_out, nx, &S_forw_result, 0, 0);
			blasfeo_pack_dvec(nx, in->S_adj, &adjoint_seed, 0);

			blasfeo_dgemv_t(nx, nx+nu, 1.0, &S_forw_result, 0, 0, &adjoint_seed, 0, 0.0, &forw_times_seed, 0, &forw_times_seed, 0);
			printf("S_forw^T * adj_seed = \n");
			blasfeo_print_exp_tran_dvec(nx+nu, &forw_times_seed, 0);

			blasfeo_free_dmat(&S_forw_result);
			blasfeo_free_dvec(&adjoint_seed);
			blasfeo_free_dvec(&forw_times_seed);
		}

        if (opts->output_z){
            printf("zn \n");
            d_print_exp_mat(1, nz, &out->zn[0], 1);
        }

        if (opts->sens_algebraic){
            printf("algebraic sensitivities \n");
            d_print_exp_mat(nz, NF, &out->S_algebraic[0], nz);
        }

		if(opts->sens_hess){
			double *S_hess = out->S_hess;
			printf("S_hess: \n");
			d_print_exp_mat(nx+nu, nx+nu, S_hess, nx+nu);
		}

    #if 0
		printf("\n");
		printf("cpt: %8.4f [ms]\n", 1000*out->info->CPUtime);
		printf("AD cpt: %8.4f [ms]\n", 1000*out->info->ADtime);

	#endif

		// printf("time split: %f ms CPU, %f ms LA, %f ms AD\n\n", cpu_time, la_time, ad_time);
		printf("\ntime for %d simulation steps: %f ms (AD time: %f ms (%5.2f%%))\n\n", nsim, 1e3*total_cpu_time, 1e3*ad_time, 1e2*ad_time/cpu_time);
		printf("time spent in integrator outside of casADi %f \n", 1e3*(total_cpu_time-ad_time));

    /* free memory */
		free(dims);
		free(sim_solver);
		free(in);
		free(out);
		free(opts);
		free(config);
	}
/* free external function */
    // implicit model
    external_function_casadi_free(&impl_ode_fun);
    external_function_casadi_free(&impl_ode_fun_jac_x_xdot);
    external_function_casadi_free(&impl_ode_fun_jac_x_xdot_u);
    external_function_casadi_free(&impl_ode_jac_x_xdot_u);
    // gnsf functions:
    external_function_casadi_free(&phi_fun);
    external_function_casadi_free(&phi_fun_jac_y);
    external_function_casadi_free(&phi_jac_y_uhat);
    external_function_casadi_free(&f_lo_fun_jac_x1k1uz);
    external_function_casadi_free(&get_matrices_fun);

	printf("\nsuccess!\n");

    return 0;
}
