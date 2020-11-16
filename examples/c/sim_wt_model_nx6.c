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

// external
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// acados
// TODO(dimitris): remove most includes
#include "acados/sim/sim_common.h"
#include "acados/sim/sim_gnsf.h"

#include "acados/utils/external_function_generic.h"

#include "acados_c/external_function_interface.h"
#include "acados_c/sim_interface.h"

// blasfeo
#include "blasfeo/blasfeo_common.h"
#include "blasfeo/blasfeo_d_aux.h"
#include "blasfeo/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/blasfeo_d_blas.h"
#include "blasfeo/blasfeo_v_aux_ext_dep.h"

// wt model
#include "examples/c/wt_model_nx6/wt_model.h"

// x0 and u for simulation
#include "examples/c/wt_model_nx6/u_x0.c"



int main()
{

	/************************************************
	* initialization
	************************************************/
    int ii, jj;

    int nx = 6;
    int nu = 2;
	int np = 1;

	int nsim = 1000;

    int NF = nx + nu; // columns of forward seed

    double Ts = 0.2; // simulation time

	double *x_sim = malloc(sizeof(double)*nx*(nsim+1));

	for (ii=0; ii<nx; ii++)
		x_sim[ii] = x_ref[ii];

	/************************************************
	* external functions (explicit model)
	************************************************/

	// expl_ode_fun
	external_function_param_casadi expl_ode_fun;
	expl_ode_fun.casadi_fun = &casadi_expl_ode_fun;
	expl_ode_fun.casadi_work = &casadi_expl_ode_fun_work;
	expl_ode_fun.casadi_sparsity_in = &casadi_expl_ode_fun_sparsity_in;
	expl_ode_fun.casadi_sparsity_out = &casadi_expl_ode_fun_sparsity_out;
	expl_ode_fun.casadi_n_in = &casadi_expl_ode_fun_n_in;
	expl_ode_fun.casadi_n_out = &casadi_expl_ode_fun_n_out;
	external_function_param_casadi_create(&expl_ode_fun, np);

	// expl_vde_for
	external_function_param_casadi expl_vde_for;
	expl_vde_for.casadi_fun = &casadi_expl_vde_for;
	expl_vde_for.casadi_work = &casadi_expl_vde_for_work;
	expl_vde_for.casadi_sparsity_in = &casadi_expl_vde_for_sparsity_in;
	expl_vde_for.casadi_sparsity_out = &casadi_expl_vde_for_sparsity_out;
	expl_vde_for.casadi_n_in = &casadi_expl_vde_for_n_in;
	expl_vde_for.casadi_n_out = &casadi_expl_vde_for_n_out;
	external_function_param_casadi_create(&expl_vde_for, np);

	// expl_vde_adj
	external_function_param_casadi expl_vde_adj;
	expl_vde_adj.casadi_fun = &casadi_expl_vde_adj;
	expl_vde_adj.casadi_work = &casadi_expl_vde_adj_work;
	expl_vde_adj.casadi_sparsity_in = &casadi_expl_vde_adj_sparsity_in;
	expl_vde_adj.casadi_sparsity_out = &casadi_expl_vde_adj_sparsity_out;
	expl_vde_adj.casadi_n_in = &casadi_expl_vde_adj_n_in;
	expl_vde_adj.casadi_n_out = &casadi_expl_vde_adj_n_out;
	external_function_param_casadi_create(&expl_vde_adj, np);

	/************************************************
	* external functions (implicit model)
	************************************************/

	// impl_ode_fun
	external_function_param_casadi impl_ode_fun;
	impl_ode_fun.casadi_fun = &casadi_impl_ode_fun;
	impl_ode_fun.casadi_work = &casadi_impl_ode_fun_work;
	impl_ode_fun.casadi_sparsity_in = &casadi_impl_ode_fun_sparsity_in;
	impl_ode_fun.casadi_sparsity_out = &casadi_impl_ode_fun_sparsity_out;
	impl_ode_fun.casadi_n_in = &casadi_impl_ode_fun_n_in;
	impl_ode_fun.casadi_n_out = &casadi_impl_ode_fun_n_out;
	external_function_param_casadi_create(&impl_ode_fun, np);

	// impl_ode_fun_jac_x_xdot
	external_function_param_casadi impl_ode_fun_jac_x_xdot;
	impl_ode_fun_jac_x_xdot.casadi_fun = &casadi_impl_ode_fun_jac_x_xdot;
	impl_ode_fun_jac_x_xdot.casadi_work = &casadi_impl_ode_fun_jac_x_xdot_work;
	impl_ode_fun_jac_x_xdot.casadi_sparsity_in = &casadi_impl_ode_fun_jac_x_xdot_sparsity_in;
	impl_ode_fun_jac_x_xdot.casadi_sparsity_out = &casadi_impl_ode_fun_jac_x_xdot_sparsity_out;
	impl_ode_fun_jac_x_xdot.casadi_n_in = &casadi_impl_ode_fun_jac_x_xdot_n_in;
	impl_ode_fun_jac_x_xdot.casadi_n_out = &casadi_impl_ode_fun_jac_x_xdot_n_out;
	external_function_param_casadi_create(&impl_ode_fun_jac_x_xdot, np);

	// impl_ode_jac_x_xdot_u
	external_function_param_casadi impl_ode_jac_x_xdot_u;
	impl_ode_jac_x_xdot_u.casadi_fun = &casadi_impl_ode_jac_x_xdot_u;
	impl_ode_jac_x_xdot_u.casadi_work = &casadi_impl_ode_jac_x_xdot_u_work;
	impl_ode_jac_x_xdot_u.casadi_sparsity_in = &casadi_impl_ode_jac_x_xdot_u_sparsity_in;
	impl_ode_jac_x_xdot_u.casadi_sparsity_out = &casadi_impl_ode_jac_x_xdot_u_sparsity_out;
	impl_ode_jac_x_xdot_u.casadi_n_in = &casadi_impl_ode_jac_x_xdot_u_n_in;
	impl_ode_jac_x_xdot_u.casadi_n_out = &casadi_impl_ode_jac_x_xdot_u_n_out;
	external_function_param_casadi_create(&impl_ode_jac_x_xdot_u, np);

	/************************************************
	* external functions (Generalized Nonlinear Static Feedback (GNSF) model)
	************************************************/
    // phi_fun
    external_function_param_casadi phi_fun;
    phi_fun.casadi_fun            = &casadi_phi_fun;
    phi_fun.casadi_work           = &casadi_phi_fun_work;
    phi_fun.casadi_sparsity_in    = &casadi_phi_fun_sparsity_in;
    phi_fun.casadi_sparsity_out   = &casadi_phi_fun_sparsity_out;
    phi_fun.casadi_n_in           = &casadi_phi_fun_n_in;
    phi_fun.casadi_n_out          = &casadi_phi_fun_n_out;
	external_function_param_casadi_create(&phi_fun, np);

    // phi_fun_jac_y
    external_function_param_casadi phi_fun_jac_y;
    phi_fun_jac_y.casadi_fun            = &casadi_phi_fun_jac_y;
    phi_fun_jac_y.casadi_work           = &casadi_phi_fun_jac_y_work;
    phi_fun_jac_y.casadi_sparsity_in    = &casadi_phi_fun_jac_y_sparsity_in;
    phi_fun_jac_y.casadi_sparsity_out   = &casadi_phi_fun_jac_y_sparsity_out;
    phi_fun_jac_y.casadi_n_in           = &casadi_phi_fun_jac_y_n_in;
    phi_fun_jac_y.casadi_n_out          = &casadi_phi_fun_jac_y_n_out;
	external_function_param_casadi_create(&phi_fun_jac_y, np);

    // phi_jac_y_uhat
    external_function_param_casadi phi_jac_y_uhat;
    phi_jac_y_uhat.casadi_fun                = &casadi_phi_jac_y_uhat;
    phi_jac_y_uhat.casadi_work               = &casadi_phi_jac_y_uhat_work;
    phi_jac_y_uhat.casadi_sparsity_in        = &casadi_phi_jac_y_uhat_sparsity_in;
    phi_jac_y_uhat.casadi_sparsity_out       = &casadi_phi_jac_y_uhat_sparsity_out;
    phi_jac_y_uhat.casadi_n_in               = &casadi_phi_jac_y_uhat_n_in;
    phi_jac_y_uhat.casadi_n_out              = &casadi_phi_jac_y_uhat_n_out;

	external_function_param_casadi_create(&phi_jac_y_uhat, np);

    // f_lo_fun_jac_x1k1uz
    external_function_param_casadi f_lo_fun_jac_x1k1uz;
    f_lo_fun_jac_x1k1uz.casadi_fun            = &casadi_f_lo_fun_jac_x1k1uz;
    f_lo_fun_jac_x1k1uz.casadi_work           = &casadi_f_lo_fun_jac_x1k1uz_work;
    f_lo_fun_jac_x1k1uz.casadi_sparsity_in    = &casadi_f_lo_fun_jac_x1k1uz_sparsity_in;
    f_lo_fun_jac_x1k1uz.casadi_sparsity_out   = &casadi_f_lo_fun_jac_x1k1uz_sparsity_out;
    f_lo_fun_jac_x1k1uz.casadi_n_in           = &casadi_f_lo_fun_jac_x1k1uz_n_in;
    f_lo_fun_jac_x1k1uz.casadi_n_out          = &casadi_f_lo_fun_jac_x1k1uz_n_out;
	external_function_param_casadi_create(&f_lo_fun_jac_x1k1uz, np);

    // get_matrices_fun
    external_function_casadi get_matrices_fun;
    get_matrices_fun.casadi_fun            = &casadi_get_matrices_fun;
    get_matrices_fun.casadi_work           = &casadi_get_matrices_fun_work;
    get_matrices_fun.casadi_sparsity_in    = &casadi_get_matrices_fun_sparsity_in;
    get_matrices_fun.casadi_sparsity_out   = &casadi_get_matrices_fun_sparsity_out;
    get_matrices_fun.casadi_n_in           = &casadi_get_matrices_fun_n_in;
    get_matrices_fun.casadi_n_out          = &casadi_get_matrices_fun_n_out;
	external_function_casadi_create(&get_matrices_fun);


	int number_sim_solvers = 4;
	int nss;
	for (nss = 0; nss < number_sim_solvers; nss++)
	{
		/************************************************
		* sim plan & config
		************************************************/

		// choose plan
		sim_solver_plan plan;
		switch (nss)
		{
			case 0:
				plan.sim_solver = ERK;
				break;

			case 1:
			case 2:
				plan.sim_solver = IRK;
				break;

			case 3:
				plan.sim_solver = GNSF;
				break;

			default :
				printf("\nnot enough sim solvers implemented!\n");
				exit(1);

		}

		// create correct config based on plan
		sim_solver_config *config = sim_config_create(plan);

		/************************************************
		* sim dims
		************************************************/

		void *dims = sim_dims_create(config);
		config->set_nx(dims, nx);
		config->set_nu(dims, nu);

		/************************************************
		* sim opts
		************************************************/

		sim_rk_opts *opts = sim_opts_create(config, dims);

	//		opts->ns = 4; // number of stages in rk integrator
	//		opts->num_steps = 5; // number of integration steps
		opts->sens_adj = true;
		opts->sens_forw = true;

		sim_gnsf_dims *gnsf_dim;

		switch (nss)
		{

			case 0:
				opts->ns = 4; // number of stages in rk integrator
				opts->num_steps = 10; // number of integration steps
				break;

			case 1:
				opts->ns = 2; // number of stages in rk integrator
				opts->num_steps = 6; // number of integration steps
				break;

			case 2:
				opts->ns = 8; // number of stages in rk integrator
				opts->num_steps = 1; // number of integration steps
				opts->jac_reuse = true; // jacobian reuse
				opts->newton_iter = 3; // number of newton iterations per integration step
				break;

			case 3://gnsf
				// set additional dimensions
				gnsf_dim = (sim_gnsf_dims *) dims; // declaration not allowed inside switch somehow
				gnsf_dim->nx = nx;
				gnsf_dim->nu = nu;
				gnsf_dim->nx1= nx;
				// gnsf_dim->nx2= 0;
				gnsf_dim->ny = 5;
				gnsf_dim->nuhat = 0;
				gnsf_dim->n_out = 1;
				gnsf_dim->nz = 0;

				// set options
				opts->ns = 8; // number of stages in rk integrator
				opts->num_steps = 1; // number of integration steps
				opts->jac_reuse = true; // jacobian reuse
				opts->newton_iter = 3; // number of newton iterations per integration step
				break;

			default:
				printf("\nnot enough sim solvers implemented!\n");
				exit(1);

		}

		/************************************************
		* sim in / out
		************************************************/

		sim_in *in = sim_in_create(config, dims);

		sim_out *out = sim_out_create(config, dims);

		in->T = Ts;
		// external functions
		switch (nss)
		{
			case 0:
			{
				sim_set_model(config, in, "expl_ode_fun", &expl_ode_fun);
				sim_set_model(config, in, "expl_vde_for", &expl_vde_for);
				sim_set_model(config, in, "expl_vde_adj", &expl_vde_adj);
				break;
			}
			case 1:
			case 2: // IRK
			{
				sim_set_model(config, in, "impl_ode_fun", &impl_ode_fun);
				sim_set_model(config, in, "impl_ode_fun_jac_x_xdot", &impl_ode_fun_jac_x_xdot);
				sim_set_model(config, in, "impl_ode_jac_x_xdot_u", &impl_ode_jac_x_xdot_u);
				break;
			}
			case 3: // gnsf
			{
				// set model funtions
				sim_set_model(config, in, "phi_fun", &phi_fun);
				sim_set_model(config, in, "phi_fun_jac_y", &phi_fun_jac_y);
				sim_set_model(config, in, "phi_jac_y_uhat", &phi_jac_y_uhat);
				sim_set_model(config, in, "f_lo_jac_x1_x1dot_u_z", &f_lo_fun_jac_x1k1uz);

				// import model matrices
				external_function_generic *get_model_matrices = (external_function_generic *) &get_matrices_fun;
				sim_gnsf_import_matrices(gnsf_dim, in->model, get_model_matrices);
				break;
			}
			default :
			{
				printf("\nnot enough sim solvers implemented!\n");
				exit(1);
			}
		}

		// seeds forw
		for (ii = 0; ii < nx * NF; ii++)
			in->S_forw[ii] = 0.0;
		for (ii = 0; ii < nx; ii++)
			in->S_forw[ii * (nx + 1)] = 1.0;

		// seeds adj
		for (ii = 0; ii < nx; ii++)
			in->S_adj[ii] = 1.0;
		for (ii = 0; ii < nu; ii++)
			in->S_adj[ii+nx] = 0.0;

		/************************************************
		* sim solver
		************************************************/

		// print solver info
		printf("\n ===  USING SOLVER NUMBER %d === \n",nss);
		switch (nss)
		{

			case 0:
				printf("\n\nsim solver: ERK, ns=%d, num_steps=%d\n", opts->ns, opts->num_steps);
				plan.sim_solver = ERK;
				break;

			case 1:
			case 2:
				printf("\n\nsim solver: IRK, ns=%d, num_steps=%d\n", opts->ns, opts->num_steps);
				plan.sim_solver = IRK;
				break;
			case 3:
				printf("\n\nsim solver: gnsf, ns=%d, num_steps=%d\n", opts->ns, opts->num_steps);
				plan.sim_solver = GNSF;
				break;

			default :
				printf("\nnot enough sim solvers implemented!\n");
				exit(1);

		}

		sim_solver *sim_solver = sim_create(config, dims, opts);

		int acados_return;

		if (nss == 3) // for gnsf: perform precomputation
			sim_gnsf_precompute(config, gnsf_dim, in->model, opts, sim_solver->mem, sim_solver->work, in->T);


		acados_timer timer;
		acados_tic(&timer);

		int nsim0 = nsim;

		double cpu_time = 0.0;
		double la_time = 0.0;
		double ad_time = 0.0;

		// to avoid unstable behavior introduce a small pi-controller for rotor speed tracking
		double uctrl = 0.0;
		double uctrlI = 0.0;
		double kI = 1e-1;
		double kP = 10;
		double tmp, ctrlErr;



		for (ii=0; ii<nsim; ii++)
		{
			// update initial state
			for (jj = 0; jj < nx; jj++)
				in->x[jj] = x_sim[ii*nx+jj];

			// compute inputs
			for (jj = 0; jj < nu; jj++)
				in->u[jj] = u_sim[ii*nu+jj];
			tmp = in->u[1] - uctrl;
			in->u[1] = tmp>0.0 ? tmp : 0.0;

			// update parameters
			switch (nss)
			{
				case 0:
				{
					expl_ode_fun.set_param(&expl_ode_fun, p_sim+ii*np);
					expl_vde_for.set_param(&expl_vde_for, p_sim+ii*np);
					expl_vde_for.set_param(&expl_vde_adj, p_sim+ii*np);
					break;
				}
				case 1:
				case 2:
				{
					impl_ode_fun.set_param(&impl_ode_fun, p_sim+ii*np);
					impl_ode_fun_jac_x_xdot.set_param(&impl_ode_fun_jac_x_xdot, p_sim+ii*np);
					impl_ode_jac_x_xdot_u.set_param(&impl_ode_jac_x_xdot_u, p_sim+ii*np);
					break;
				}
				case 3:
				{
					phi_fun.set_param(&phi_fun, p_sim+ii*np);
					phi_fun_jac_y.set_param(&phi_fun_jac_y, p_sim+ii*np);
					phi_jac_y_uhat.set_param(&phi_jac_y_uhat, p_sim+ii*np);
					f_lo_fun_jac_x1k1uz.set_param(&f_lo_fun_jac_x1k1uz, p_sim+ii*np);
					break;
				}
				default :
				{
					printf("\nnot enough sim solvers implemented!\n");
					exit(1);
				}
			}

			// d_print_mat(1, nx, in->x, 1);
			// d_print_mat(1, nu, in->u, 1);

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
			for (jj = 0; jj < nx; jj++)
				x_sim[(ii+1)*nx+jj] = out->xn[jj];

			// update PI-controller
			ctrlErr = x_ref[nx*(ii+1)] - x_sim[nx*(ii+1)];
			uctrlI = uctrlI + kI*ctrlErr*Ts;
			uctrl = kP*ctrlErr + uctrlI;

			// if (ii < nsim-1)
			// 	printf("\nii = %d, sim error = %e\n", ii, ctrlErr);
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
			printf("\nS_forw_out: \n");
			d_print_exp_mat(nx, NF, S_forw_out, nx);
		}

		if(opts->sens_adj){
			double *S_adj_out = out->S_adj;
			printf("\nS_adj_out: \n");
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

    #if 0
		printf("\n");
		printf("cpt: %8.4f [ms]\n", 1000*out->info->CPUtime);
		printf("AD cpt: %8.4f [ms]\n", 1000*out->info->ADtime);

	#endif

		// printf("time split: %f ms CPU, %f ms LA, %f ms AD\n\n", cpu_time, la_time, ad_time);
		printf("\ntime for %d simulation steps: %f ms (AD time: %f ms (%5.2f%%))\n\n", nsim, 1e3*total_cpu_time, 1e3*ad_time, 1e2*ad_time/cpu_time);
		printf("time spent in integrator outside of casADi %f \n", 1e3*(total_cpu_time-ad_time));

		/************************************************
		* free memory
		************************************************/
		free(dims);
		free(sim_solver);
		free(in);
		free(out);
		free(opts);
		free(config);
	}

	free(x_sim);

	// explicit model
	external_function_param_casadi_free(&expl_ode_fun);
	external_function_param_casadi_free(&expl_vde_for);
	external_function_param_casadi_free(&expl_vde_adj);
	// implicit model
	external_function_param_casadi_free(&impl_ode_fun);
	external_function_param_casadi_free(&impl_ode_fun_jac_x_xdot);
	external_function_param_casadi_free(&impl_ode_jac_x_xdot_u);
	// gnsf functions:
	external_function_param_casadi_free(&f_lo_fun_jac_x1k1uz);
	external_function_param_casadi_free(&phi_fun);
	external_function_param_casadi_free(&phi_fun_jac_y);
	external_function_param_casadi_free(&phi_jac_y_uhat);

	external_function_casadi_free(&get_matrices_fun);
	printf("\nsuccess!\n\n");

    return 0;
}
