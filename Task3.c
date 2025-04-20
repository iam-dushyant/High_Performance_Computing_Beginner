/*******************************************************************************
2D advection example program which advects a Gaussian u(x,y) at a fixed velocity



Outputs: initial.dat - inital values of u(x,y) 
         final.dat   - final values of u(x,y)

         The output files have three columns: x, y, u

         Compile with: gcc -o advection2D -std=c99 advection2D.c -lm

Notes: The time step is calculated using the CFL condition

********************************************************************************/

/*********************************************************************
                     Include header files 
**********************************************************************/

#include <stdio.h>
#include <math.h>
#include <omp.h>

/*********************************************************************
                      Main function
**********************************************************************/

int main(){

  /* Grid properties */
  const int NX=1000;    // Number of x points
  const int NY=1000;    // Number of y points
  const float xmin=0.0; // Minimum x value
  const float xmax=30.0; // Maximum x value
  const float ymin=0.0; // Minimum y value
  const float ymax=30.0; // Maximum y value
  
  /* Parameters for the Gaussian initial conditions */
  const float t0=3.0;                    // Centre(x)
  const float y0=15.0;                   // Centre(y)
  const float z0=1.0;
  const float sigmat=1.0;               // Width(x)
  const float sigmay=5.0;               // Width(y)
  const float sigmat2 = sigmat * sigmat; // Width(x) squared
  const float sigmay2 = sigmay * sigmay; // Width(y) squared

  /* Boundary conditions */
  const float bval_left=0.0;    // Left boundary value
  const float bval_right=0.0;   // Right boundary value
  const float bval_lower=0.0;   // Lower boundary
  const float bval_upper=0.0;   // Upper boundary
  
  /* Time stepping parameters */
  const float CFL=0.9;   // CFL number 
  const int nsteps=1000; // Number of time steps

  /* Velocity */
  float velx[NX+1];     // velocity as a gradient
  const float vely = 0.0;
  const float ustar = 0.1;
  const float kappa = 0.41;
  
  /* Arrays to store variables. These have NX+2 elements
     to allow boundary values to be stored at both ends */
  float x[NX+2];          // x-axis values
  float y[NY+2];          // y-axis values
  float t[NX+2];
  float u[NX+2][NY+2];    // Array of u values
  float dudt[NX+2][NY+2]; // Rate of change of u

  float t2;   // x squared (used to calculate iniital conditions)
  float y2;   // y squared (used to calculate iniital conditions)
  
  /* Calculate distance between points */
  float dx = (xmax-xmin) / ( (float) NX);
  float dy = (ymax-ymin) / ( (float) NY);
  
  /* Calculate time step using the CFL condition */
  /* The fabs function gives the absolute value in case the velocity is -ve */
  float dt = CFL * dx;
  
  /*** Report information about the calculation ***/
  printf("Grid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("Time step           = %g\n", dt);
  printf("No. of time steps   = %d\n", nsteps);
  printf("End time            = %g\n", dt*(float) nsteps);

  /*** Place x points in the middle of the cell ***/
  /* LOOP 1 */
  #pragma omp parallel for shared(dx) private(x)
  for (int i=0; i<NX+2; i++){
    x[i] = ( (float) i - 0.5) * dx;
  }

  /*** Place y points in the middle of the cell ***/
  /* LOOP 2 */
  #pragma omp parallel for shared(dy) private(y)
  for (int j=0; j<NY+2; j++){
    y[j] = ( (float) j - 0.5) * dy;
  }

  /*** Creating t as a function to further use in calculating u ***/
  #pragma omp parallel for shared(dt) private(t)
  for (int k=0; k<NY+2; k++){
    t[k] = ( (float) k - 0.5) * dt;
  }

  /*** Loop to calculate the velocity as a function of horizontal velocity ***/
  #pragma omp parallel for shared(y) private(velx)
  for (int i=0; i<NX+1; i++) {
    if (y[i] <= z0) {
      velx[i] = 0;
    }
    else if (y[i] > z0) {
      velx[i] = (ustar/kappa) * log(y[i]/z0);
    }
    }

  /*** Set up Gaussian initial conditions ***/
  /* LOOP 3 */
//#pragma omp parallel for collapse(2)  private(x2, y2)
//  for (int i=0; i<NX+2; i++){
//    for (int j=0; j<NY+2; j++){
//      x2      = (x[i]-x0) * (x[i]-x0);
//      y2      = (y[j]-y0) * (y[j]-y0);
//      u[i][j] = exp( -1.0 * ( (x2/(2.0*sigmax2)) + (y2/(2.0*sigmay2)) ) );
//    }
//  }

  /*** Set up initial conditions ***/
  #pragma omp parallel for collapse(2)
   for (int i=0; i<NX+1; i++){
     for (int j=0; j<NY+1; j++){
       u[i][j] = 0;
    }
   }

  /*** Write array of initial u values out to file ***/
  FILE *initialfile;
  initialfile = fopen("initial_task3.dat", "w");
  /* LOOP 4 */
  //PARALLELISATION IS NOT REQUIRED AS u IS INTERDEPENDENT ON BOTH LOOPS i, j.
  //THREADING WOULD TAKE PLACE IN NO SPECIFIC ORDER, AND THEREFORE, WRITING
  //THESE TO A FILE WOULD YIELD TO A NaN VALUE IN MOST LINES WHEN WRITING
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(initialfile);
    
    /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
    #pragma omp parallel for collapse(2) shared(y, t) private(y2, t2)
    for (int i=0; i<NX+1; i++){
     for (int j=0; j<NY+1; j++){
       y2 = (y[j] - y0) * (y[j] - y0);
       t2 = (t[i] - t0) * (t[i] - t0);
       u[i][j] = exp(-1.0 * ((y2 / (2 * sigmay2)) + (t2 / (2 * sigmat2))));
     }
    }

    /*** Update solution by looping over time steps ***/
    /* LOOP 5 */
    //PARALLELISATION FOR THIS LOOP IS NOT HELPFUL, AS IT WOULD JUST ESSENTIALLY
    //CREATE A NEW TASK FOR THE INNER LOOPS TO PERFORM COMPUTATION. IT IS, INSTEAD,
    //EFFICIENT TO PARALLELISE INNER LOOPS.

  for (int m=0; m<nsteps; m++){

    /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
    /* LOOP 6 */
//#pragma omp parallel for shared(u)
//    for (int j=0; j<NY+2; j++){
//      u[0][j]    = bval_left;
//      u[NX+1][j] = bval_right;
//    }

    /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
    /* LOOP 7 */
//#pragma omp parallel for shared(u)
//    for (int i=0; i<NX+2; i++){
//      u[i][0]    = bval_lower;
//      u[i][NY+1] = bval_upper;
//    }

    /*** Calculate rate of change of u using leftward difference ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 8 */
    #pragma omp parallel for collapse(2) shared(velx, u) private(dudt, dx, dy)
    for (int i=0; i<NX+1; i++){
      for (int j=0; j<NY+1; j++){
	dudt[i][j] = -velx[j] * (u[i][j] - u[i-1][j]) / dx - vely * (u[i][j] - u[i][j-1]) / dy;
      }
    }

    /*** Update u from t to t+dt ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 9 */
    #pragma omp parallel for collapse(2) shared(dudt) private(u)
    for	(int i=0; i<NX+1; i++){
      for (int j=0; j<NY+1; j++){
	u[i][j] = u[i][j] + dudt[i][j] * dt;
      }
    }
  } // time loop
  
  /*** Write array of final u values out to file ***/
  FILE *finalfile;
  finalfile = fopen("final_task3.dat", "w");
  /* LOOP 10 */
  //PARALLELISATION IS NOT REQUIRED AS u IS INTERDEPENDENT ON BOTH LOOPS i, j.
  //THREADING WOULD TAKE PLACE IN NO SPECIFIC ORDER, AND THEREFORE, WRITING
  //THESE TO A FILE WOULD NOT YIELD TO ACCURATE RESULTS
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(finalfile);

  return 0;
}



/* End of file ******************************************************/
