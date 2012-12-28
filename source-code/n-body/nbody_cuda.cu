#include "util/cuPrintf.cu"
#include <stdio.h>



__global__ void kernel( 
	double* x, double* y, double* z,
	double* xnew, double* ynew, double* znew,
	double* vx, double* vy, double* vz,
	double* m)
{
	int gloablId = blockIdx.x * blockDim.x + threadIdx.x;
	//cuPrintf("Hello, world from the device!%d\n",gridDim.y);
	
	int j;
	
	double ax = 0.0;
	double ay = 0.0;
	double az = 0.0;
	double dx = 0.0;
	double dy = 0.0;
	double dz = 0.0;
	double distSqr = 0.0;
	double distSixth = 0.0;
	double f = 0.0; 

	double dt = 0.1;
        double eps = 0.1;	
	
	for(j=0; j<(blockDim.x * gridDim.x); j++) { 
		dx = x[j] - x[gloablId];			
		dy = y[j] - y[gloablId];
		dz = z[j] - z[gloablId];

		distSqr = 1.0/sqrt(dx * dx + dy * dy + dz * dz + eps);  
		distSixth = distSqr * distSqr * distSqr;
				
		f = m[j] * distSixth; 

		ax += f * dx;		
		ay += f * dy;
		az += f * dz;

	}
	
	
	xnew[gloablId] = x[gloablId] + dt*vx[gloablId] + 0.5*dt*dt*ax;
	ynew[gloablId] = y[gloablId] + dt*vy[gloablId] + 0.5*dt*dt*ay;
	znew[gloablId] = z[gloablId] + dt*vz[gloablId] + 0.5*dt*dt*az;
	//cuPrintf("Hello, world from the device!%d\n",x[gloablId]);
	

	vx[gloablId] += dt*ax;
	vy[gloablId] += dt*ay;
	vz[gloablId] += dt*az;

  	
}


int main(void)
{



	int nparticle = 100; 
	int nthread = 10; 

	int problemSizeX      = nparticle;

	double num_bytes = nparticle * sizeof(double);

	double* deviceX = 0;
	double* deviceY = 0;
	double* deviceZ = 0;
	double* deviceXnew = 0;
	double* deviceYnew = 0;
	double* deviceZnew = 0;
	double* deviceVX = 0;
	double* deviceVY = 0;
	double* deviceVZ = 0;
	double* deviceM = 0;

	double* hostX = 0;
	double* hostY = 0;
	double* hostZ = 0;
	double* hostXnew = 0;
	double* hostYnew = 0;
	double* hostZnew = 0;
	double* hostVX = 0;
	double* hostVY = 0;
	double* hostVZ = 0;
	double* hostM = 0;

	// allocate memory in either space
	hostX = (double*)malloc(num_bytes);
	hostY = (double*)malloc(num_bytes);
	hostZ = (double*)malloc(num_bytes);

	hostXnew = (double*)malloc(num_bytes);
	hostYnew = (double*)malloc(num_bytes);
	hostZnew = (double*)malloc(num_bytes);

	hostVX = (double*)malloc(num_bytes);
	hostVY = (double*)malloc(num_bytes);
	hostVZ = (double*)malloc(num_bytes);

	hostM = (double*)malloc(num_bytes);

	cudaMalloc((void**)&deviceX, num_bytes);
	cudaMalloc((void**)&deviceY, num_bytes);
	cudaMalloc((void**)&deviceZ, num_bytes);

	cudaMalloc((void**)&deviceXnew, num_bytes);
	cudaMalloc((void**)&deviceYnew, num_bytes);
	cudaMalloc((void**)&deviceZnew, num_bytes);

	cudaMalloc((void**)&deviceVX, num_bytes);
	cudaMalloc((void**)&deviceVY, num_bytes);
	cudaMalloc((void**)&deviceVZ, num_bytes);

	cudaMalloc((void**)&deviceM, num_bytes); 

	for (int i = 0; i < problemSizeX; ++i) {
		hostX[i] = 0.7 + i;
		hostY[i] = 0.8 + i;
		hostZ[i] = 0.9 + i;

		hostVX[i] = 0.0;
		hostVY[i] = 0.0;
		hostVZ[i] = 0.0;

		hostXnew[i] = 0.0;
		hostYnew[i] = 0.0;
		hostZnew[i] = 0.0;

		hostM[i] =  0.5 + i;
	}
	printf("%f\n",hostX[0]);




	cudaMemset(deviceX, 0, num_bytes);
	cudaMemset(deviceY, 0, num_bytes);
	cudaMemset(deviceZ, 0, num_bytes);

	cudaMemset(deviceXnew, 0, num_bytes);
	cudaMemset(deviceYnew, 0, num_bytes);
	cudaMemset(deviceZnew, 0, num_bytes);

	cudaMemset(deviceVX, 0, num_bytes);
	cudaMemset(deviceVY, 0, num_bytes);
	cudaMemset(deviceVZ, 0, num_bytes);

	cudaMemset(deviceM, 0, num_bytes);

	// create two dimensional 4x4 thread blocks
	dim3 block_size;
	block_size.x = nthread;

	// configure a two dimensional grid as well
	dim3 grid_size;
	grid_size.x = nparticle / nthread;


	cudaMemcpy(deviceX, hostX, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceY, hostY, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceZ, hostZ, num_bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(deviceXnew, hostXnew, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceYnew, hostYnew, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceZnew, hostZnew, num_bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(deviceVX, hostVX, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceVY, hostVY, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceVZ, hostVZ, num_bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(deviceM, hostM, num_bytes, cudaMemcpyHostToDevice);

	//double deviceStart = getTimeStamp();

	cudaPrintfInit();
	printf("%f\n",hostX[20]);
	kernel<<<grid_size,block_size>>>(deviceX, deviceY, deviceZ, deviceXnew, deviceYnew, deviceZnew, deviceVX, deviceVY, deviceVZ, deviceM);
	//kernel<<<10,10>>>(deviceX, deviceY, deviceZ, deviceXnew, deviceYnew, deviceZnew, deviceVX, deviceVY, deviceVZ, deviceM);
	//printf("%f\n",hostX[0]);
	cudaPrintfDisplay();
	
	cudaPrintfEnd();
	//double deviceEnd = getTimeStamp();

	//std::cout << "Device Time:  " << (deviceEnd - deviceStart) << "s\n";

	// download and inspect the result on the host:
	cudaMemcpy(hostX, deviceXnew, num_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostY, deviceYnew, num_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostZ, deviceZnew, num_bytes, cudaMemcpyDeviceToHost);

	cudaMemcpy(hostVX, deviceVX, num_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostVY, deviceVY, num_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostVZ, deviceVZ, num_bytes, cudaMemcpyDeviceToHost);


	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		//exit(-1);
	}
	printf("Cuda %f\n",hostX[20]);

	// deallocate memory
	free(hostX);
	free(hostY);
	free(hostZ);

	free(hostXnew);
	free(hostYnew);
	free(hostZnew);

	free(hostVX);
	free(hostVY);
	free(hostVZ);

	free(hostM);

	cudaFree(deviceX);
	cudaFree(deviceY);
	cudaFree(deviceZ);

	cudaFree(deviceXnew);
	cudaFree(deviceYnew);
	cudaFree(deviceZnew);

	cudaFree(deviceVX);
	cudaFree(deviceVY);
	cudaFree(deviceVZ);

	cudaFree(deviceM);




  	int n = nparticle;
	//double num_bytes = n * sizeof(double);

	double *x =  0; 
	double *y =  0;
	double *z =  0;

	double *xnew =  0; 
	double *ynew =  0;
	double *znew =  0;

	double *m = 0; 
	double *vx = 0; 
	double *vy = 0;
	double *vz =  0; 

	x = (double*)malloc(num_bytes);
	y = (double*)malloc(num_bytes);
	z = (double*)malloc(num_bytes);

	xnew = (double*)malloc(num_bytes);
	ynew = (double*)malloc(num_bytes);
	znew = (double*)malloc(num_bytes);

	vx = (double*)malloc(num_bytes);
	vy = (double*)malloc(num_bytes);
	vz = (double*)malloc(num_bytes);

	m = (double*)malloc(num_bytes);

	for(int i=0; i<n; i++) {
		x[i] = 0.7+i;
		y[i] = 0.8+i;
		z[i] = 0.9+i;

		m[i] = 0.5+i;


		vx[i] = 0.0;
		vy[i] = 0.0;
		vz[i] = 0.0;

		xnew[i] = 0.0;
		ynew[i] = 0.0;
		znew[i] = 0.0;
	}
	


	double eps = 0.1;
	double dt =0.1;
	//printf("%f\n",xnew[0]);
	for(int i=0; i<n; i++) { /* Foreach particle "i" ... */
		double ax=0.0;
		double ay=0.0;
		double az=0.0;

		double dx=0.0;
		double dy=0.0;
		double dz=0.0;
		double invr=0.0;
		double invr3=0.0;
		double f=0.0;

	      	for(int j=0; j<n; j++) { /* Loop over all particles "j" */
			dx=x[j]-x[i];
			dy=y[j]-y[i];
			dz=z[j]-z[i];

			invr = 1.0/sqrt(dx*dx + dy*dy + dz*dz + eps);
	       		invr3 = invr*invr*invr;

	      		f=m[j]*invr3;
	       		ax += f*dx; /* accumulate the acceleration from gravitational attraction */
	       		ay += f*dy;
	       		az += f*dx;
			//printf("%f\n",ax);
	     	}
		xnew[i] = x[i] + dt*vx[i] + 0.5*dt*dt*ax; /* update position of particle "i" */
		ynew[i] = y[i] + dt*vy[i] + 0.5*dt*dt*ay;
		znew[i] = z[i] + dt*vz[i] + 0.5*dt*dt*az;
		//prdoublef("%d\n",xnew[i]);
	}
	//prdoublef("%d\n",xnew[0]);
	for(int i=0;i<n;i++) { /* copy updated positions back doubleo original arrays */
		x[i] = xnew[i];
		y[i] = ynew[i];
		z[i] = znew[i];
	}

	printf("%f\n",x[20]);

	//free(x);

  return 0;
}
