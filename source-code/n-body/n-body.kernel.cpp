/*
 * Copyright (C) 2011 by Justin Holewinski
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include<math.h>
extern "C"
void n_body(float* x, float* y, float* z,
	float* xnew, float* ynew, float* znew,
	float* vx, float* vy, float* vz,
	float* m) {

  	// Determine our global offset floato the vector
  	int globalId = (__builtin_ptx_read_ctaid_x() * __builtin_ptx_read_ntid_x())
    		+ __builtin_ptx_read_tid_x();


	int j;
	
	float ax = 0.0f;
	float ay = 0.0f;
	float az = 0.0f;
	float dx = 0.0f;
	float dy = 0.0f;
	float dz = 0.0f;
	float distSqr = 0.0f;
	float distSixth = 0.0f;
	float f = 0.0f; 

	float dt = 0.1f;
        float eps = 0.1f;	
	int k = __builtin_ptx_read_ntid_x() * __builtin_ptx_read_nctaid_x();	

	for(j=0; j<k; j++) { 
		dx = x[j] - x[globalId];			
		dy = y[j] - y[globalId];
		dz = z[j] - z[globalId];

		distSqr = 1.0f/sqrt(dx * dx + dy * dy + dz * dz + eps);  
		distSixth = distSqr * distSqr * distSqr;
				
		f = m[j] * distSixth; 

		ax += f * dx;		
		ay += f * dy;
		az += f * dz;

	}
	
	
	xnew[globalId] = x[globalId] + dt*vx[globalId] + 0.5f*dt*dt*ax;
	ynew[globalId] = y[globalId] + dt*vy[globalId] + 0.5f*dt*dt*ay;
	znew[globalId] = z[globalId] + dt*vz[globalId] + 0.5f*dt*dt*az;
	//cuPrintf("Hello, world from the device!%d\n",x[globalId]);
	

	vx[globalId] += dt*ax;
	vy[globalId] += dt*ay;
	vz[globalId] += dt*az;

	


}
