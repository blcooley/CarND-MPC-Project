#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 10;
double dt = 0.1;

size_t x_start =     N * 0;
size_t y_start =     N * 1;
size_t psi_start =   N * 2;
size_t v_start =     N * 3;
size_t cte_start =   N * 4;
size_t epsi_start =  N * 5;
size_t delta_start = N * 6;
size_t a_start =     N * 7 - 1; // Only N - 1 delta entries

double ref_v = 50;

double cte_cost = 1000.0;
double epsi_cost = 1000.0;
double v_cost = 1.0;
double delta_cost = 555.0;
double a_cost = 1.5;
double ddelta_cost = 100.0;
double da_cost = 10.0;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    // The following implementation is derived from Lesson 20, Lecture 9 of SDCND, 2018/02/19
    
    // The cost is stored is the first element of `fg`.
    // Any additions to the cost should be added to `fg[0]`.

    fg[0] = 0;
    for (size_t t = 0; t < N; t++) {
      fg[0] += cte_cost * CppAD::pow(vars[cte_start+t],2);
      fg[0] += epsi_cost * CppAD::pow(vars[epsi_start+t],2);
      fg[0] += v_cost * CppAD::pow(vars[v_start+t]-ref_v,2);
    }

    // Minimize the use of actuators.
    for(size_t t = 0; t < N-1; t++) {
      fg[0] += delta_cost * CppAD::pow(vars[delta_start + t] * vars[v_start+t], 2);
      fg[0] += a_cost * CppAD::pow(vars[a_start + t], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (size_t t = 0; t < N-2; t++) {
      fg[0] += ddelta_cost * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += da_cost * CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    fg[x_start + 1]    = vars[x_start];
    fg[y_start + 1]    = vars[y_start];
    fg[psi_start + 1]  = vars[psi_start];
    fg[v_start + 1]    = vars[v_start];
    fg[cte_start + 1]  = vars[cte_start];
    fg[epsi_start + 1] = vars[epsi_start];


    for (size_t t = 0; t < N-1; ++t) {
      AD<double> x1    = vars[x_start + t + 1];
      AD<double> y1    = vars[y_start + t + 1];
      AD<double> psi1  = vars[psi_start + t + 1];
      AD<double> v1    = vars[v_start + t + 1];
      AD<double> cte1  = vars[cte_start + t + 1];
      AD<double> epsi1 = vars[epsi_start + t + 1];

  
      AD<double> x0 = vars[x_start + t];
      AD<double> y0 = vars[y_start + t];
      AD<double> psi0 = vars[psi_start + t];
      AD<double> v0  = vars[v_start + t];
      AD<double> delta0 = vars[delta_start + t];
      AD<double> cte0 = vars[cte_start + t];
      AD<double> epsi0 = vars[epsi_start + t];
      AD<double> a0 = vars[a_start + t];

      if (t > 1) {
	// Use t-2 timetamp to Account to latancy 
        a0     = vars[a_start + t - 1];
        delta0 = vars[delta_start + t - 1];
      }

      AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * CppAD::pow(x0, 2) + coeffs[3] * CppAD::pow(x0, 3);
      AD<double> psides0 = CppAD::atan(coeffs[1] + 2 * coeffs[2] * x0 + 3 * coeffs[3] * CppAD::pow(x0, 2));

      fg[x_start + t + 2]    = x1 - (x0 + v0 *  CppAD::cos(psi0) * dt);
      fg[y_start + t + 2]    = y1 - (y0 + v0 *  CppAD::sin(psi0) * dt);
      fg[psi_start + t + 2]  = psi1 - (psi0 - v0 * (delta0/Lf) * dt);
      fg[v_start + t + 2]    = v1 - (v0 + a0 * dt);
      fg[cte_start + t + 2]  = cte1 - ((f0 - y0) + v0 * CppAD::sin(epsi0) * dt);
      fg[epsi_start + t + 2] = epsi1 - ((psi0 - psides0) - v0 * (delta0 / Lf) * dt);

    }
  }
};
  
//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9
  size_t n_vars = N * 6 + (N - 1) * 2;
  // TODO: Set the number of constraints
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  // Set initial state values
  vars[x_start] =    state[0];
  vars[y_start] =    state[1];
  vars[psi_start] =  state[2];
  vars[v_start] =    state[3];
  vars[cte_start] =  state[4];
  vars[epsi_start] = state[5];

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.
  for (size_t i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -10000000000.0;
    vars_upperbound[i] =  10000000000.0;
  }

  // Bounds for steering are -25 degrees to 25 degrees => -0.43633 to 0.43633 radians
  for(size_t i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.43633;
    vars_upperbound[i] =  0.43633;
  }

  // Throttle between -1 and 1
  for(size_t i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1;
    vars_upperbound[i] =  1;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (size_t i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  // Constrain initial values to the state values
  constraints_lowerbound[x_start] =    state[0];
  constraints_upperbound[x_start] =    state[0];
  constraints_lowerbound[y_start] =    state[1];
  constraints_upperbound[y_start] =    state[1];
  constraints_lowerbound[psi_start] =  state[2];
  constraints_upperbound[psi_start] =  state[2];
  constraints_lowerbound[v_start] =    state[3];
  constraints_upperbound[v_start] =    state[3];
  constraints_lowerbound[cte_start] =  state[4];
  constraints_upperbound[cte_start] =  state[4];
  constraints_lowerbound[epsi_start] = state[5];
  constraints_upperbound[epsi_start] = state[5];

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  std::vector<double> result {solution.x[delta_start], solution.x[a_start]};
  for (size_t t = 0; t < N - 1; ++t) {
    result.push_back(solution.x[x_start + t + 1]);
  }

  for (size_t t = 0; t < N - 1; ++t) {
    result.push_back(solution.x[y_start + t + 1]);
  }

  return result;
}
