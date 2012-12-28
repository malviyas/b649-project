

var n = 100;
var problemSize = n;

var x =  new ArrayBuffer(problemSize*8); 
var y =  new ArrayBuffer(problemSize*8); 
var z =  new ArrayBuffer(problemSize*8); 

var xnew =  new ArrayBuffer(problemSize*8); 
var ynew =  new ArrayBuffer(problemSize*8); 
var znew =  new ArrayBuffer(problemSize*8); 

var m = new ArrayBuffer(problemSize*8); 
var vx =  new ArrayBuffer(problemSize*8); 
var vy =  new ArrayBuffer(problemSize*8); 
var vz =  new ArrayBuffer(problemSize*8); 


var float32ViewX = new Float32Array(x);
var float32ViewY = new Float32Array(y);
var float32ViewZ = new Float32Array(z);

var float32ViewXnew = new Float32Array(xnew);
var float32ViewYnew = new Float32Array(ynew);
var float32ViewZnew = new Float32Array(znew);

var float32ViewVX = new Float32Array(vx);
var float32ViewVY = new Float32Array(vy);
var float32ViewVZ = new Float32Array(vz);

var float32ViewM = new Float32Array(m);

for(i=0; i<n; i++) {
/*	float32ViewX[i] = Math.floor((Math.random()*100)+1);
	float32ViewY[i] = Math.floor((Math.random()*100)+1);
	float32ViewZ[i] = Math.floor((Math.random()*100)+1);

	float32ViewM[i] = Math.floor((Math.random()*100)+1);

	float32ViewVX[i] = 0;
	float32ViewVY[i] = 0;
	float32ViewVZ[i] = 0;
*/
	float32ViewX[i] = 0.7+i;
	float32ViewY[i] = 0.8+i;
	float32ViewZ[i] = 0.9+i;

	float32ViewM[i] = 0.5+i;


	float32ViewVX[i] = 0.0;
	float32ViewVY[i] = 0.0;
	float32ViewVZ[i] = 0.0;

	float32ViewXnew[i] = 0.0;
	float32ViewYnew[i] = 0.0;
	float32ViewZnew[i] = 0.0;
}

//var milliseconds1 = new Date().getTime();


//kernel
function nbody_kernel(fp_x, fp_y, fp_z, fp_xnew, fp_ynew, fp_znew, fp_vx, fp_vy, fp_vz, fp_m){

	var i_gloablId = blockIdx.x * blockDim.x + threadIdx.x;

	var f_ax = 0.0;
	var f_ay = 0.0;
	var f_az = 0.0;
	var f_dx = 0.0;
	var f_dy = 0.0;
	var f_dz = 0.0;
	var f_distSqr = 0.0;
	var f_distSixth = 0.0;
	var f_f = 0.0;

	var f_dt = 0.1;
       var f_eps = 0.1;	

	var i_j;

	for(i_j=0; i_j<(blockDim.x * gridDim.x); i_j++) { 
		f_dx = fp_x[i_j] - fp_x[gloablId];			
		f_dy = fp_y[i_j] - fp_y[gloablId];
		f_dz = fp_z[i_j] - fp_z[gloablId];

		f_distSqrt = 1.0/sqrt(f_dx * f_dx + f_dy * f_dy + f_dz * f_dz + f_eps);  
		f_distSixth = f_distSqrt * f_distSqrt * f_distSqrt;
		
		f_f = m[i_j] * f_distSixth; 

		f_ax += f_f * f_dx;		
		f_ay += f_f * f_dy;
		f_az += f_f * f_dz;

	}

	fp_xnew[gloablId] = fp_x[gloablId] + f_dt*fp_vx[gloablId] + 0.5*f_dt*f_dt*d_ax;
	fp_ynew[gloablId] = fp_y[gloablId] + f_dt*fp_vy[gloablId] + 0.5*f_dt*f_dt*d_ay;
	fp_znew[gloablId] = fp_z[gloablId] + f_dt*fp_vz[gloablId] + 0.5*f_dt*f_dt*d_az;
		
	fp_vx[gloablId] += f_dt*f_ax;
	fp_vy[gloablId] += f_dt*f_ay;
	fp_vz[gloablId] += f_dt*f_az;

}


nbody_kernel(float32ViewX.buffer, float32ViewY.buffer, float32ViewZ.buffer, float32ViewXnew.buffer, float32ViewYnew.buffer, float32ViewZnew.buffer,
				float32ViewVX.buffer,float32ViewVY.buffer,float32ViewVZ.buffer, float32ViewM.buffer);




