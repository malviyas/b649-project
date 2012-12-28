
var nProblemSize = 50000; 
var problemSize = nProblemSize;

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

for(var i=0; i<nProblemSize; i++) {
	float32ViewX[i] = 0.7+i;
	float32ViewY[i] = 0.8+i;
	float32ViewZ[i] = 0.9+i;

	float32ViewM[i] = 0.5+i;


	float32ViewVX[i] = 0.2+i;
	float32ViewVY[i] = 0.3+i;
	float32ViewVZ[i] = 0.1+i;

}
alert(float32ViewX[0]);

//alert(float32ViewXnew[0]);

var kernel_PTX = "	.version 2.0\n\
	.target compute_10, map_f64_to_f32\n\
.entry n_body (.param .b64 __param_1, .param .b64 __param_2, .param .b64 __param_3, .param .b64 __param_4, .param .b64 __param_5, .param .b64 __param_6, .param .b64 __param_7, .param .b64\n\ __param_8, .param .b64 __param_9, .param .b64 __param_10) // @vector_add\n\
{\n\
	.reg .pred %p<2>;\n\
	.reg .b32 %r<11>;\n\
	.reg .b64 %rd<40>;\n\
	.reg .f32 %f<54>;\n\
	.reg .f64 %fd<4>;\n\
// BB#0:\n\
	mov.u32	%r5, %ctaid.x;\n\
	mov.u32	%r0, %ntid.x;\n\
	mul.lo.u32	%r6, %r0, %r5;\n\
	mov.u32	%r7, %tid.x;\n\
	add.u32	%r8, %r6, %r7;\n\
	mov.u32	%r1, %nctaid.x;\n\
	mul.lo.u32	%r9, %r1, %r0;\n\
	ld.param.u64	%rd21, [__param_10];\n\
	ld.param.u64	%rd20, [__param_9];\n\
	ld.param.u64	%rd19, [__param_8];\n\
	ld.param.u64	%rd18, [__param_7];\n\
	ld.param.u64	%rd17, [__param_6];\n\
	ld.param.u64	%rd16, [__param_5];\n\
	ld.param.u64	%rd15, [__param_4];\n\
	ld.param.u64	%rd14, [__param_3];\n\
	ld.param.u64	%rd13, [__param_2];\n\
	cvt.s64.s32	%rd0, %r8;\n\
	shl.b64	%rd22, %rd0, 2;\n\
	ld.param.u64	%rd12, [__param_1];\n\
	add.u64	%rd23, %rd12, %rd22;\n\
	ld.global.f32	%f0, [%rd23];\n\
	setp.gt.s32	%p0, %r9, 0;\n\
@%p0	bra	$L__BB0_2;\n\
// BB#1:                                // %._crit_edge9\n\
	add.u64	%rd1, %rd14, %rd22;\n\
	mov.f32	%f12, 0D0000000000000000;\n\
	mov.u64	%rd39, %rd1;\n\
	mov.f32	%f51, %f12;\n\
	mov.f32	%f52, %f12;\n\
	mov.f32	%f53, %f12;\n\
	bra	$L__BB0_4;\n\
$L__BB0_2:                              // %.lr.ph\n\
	add.u64	%rd26, %rd13, %rd22;\n\
	add.u64	%rd2, %rd14, %rd22;\n\
	mov.f32	%f13, 0D0000000000000000;\n\
	ld.global.f32	%f2, [%rd2];\n\
	ld.global.f32	%f1, [%rd26];\n\
	mov.f64	%fd2, 0D3FF0000000000000;\n\
	mov.u64	%rd35, %rd12;\n\
	mov.u64	%rd36, %rd13;\n\
	mov.u64	%rd37, %rd14;\n\
	mov.u64	%rd38, %rd21;\n\
	mov.u32	%r10, %r9;\n\
	mov.f32	%f48, %f13;\n\
	mov.f32	%f49, %f13;\n\
	mov.f32	%f50, %f13;\n\
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1\n\
	mov.u64	%rd3, %rd35;\n\
	mov.u64	%rd4, %rd36;\n\
	mov.u64	%rd5, %rd37;\n\
	mov.u64	%rd6, %rd38;\n\
	mov.u32	%r3, %r10;\n\
	mov.f32	%f3, %f48;\n\
	mov.f32	%f4, %f49;\n\
	mov.f32	%f5, %f50;\n\
	ld.global.f32	%f14, [%rd5];\n\
	sub.rn.f32	%f15, %f14, %f2;\n\
	ld.global.f32	%f16, [%rd3];\n\
	sub.rn.f32	%f17, %f16, %f0;\n\
	ld.global.f32	%f18, [%rd4];\n\
	sub.rn.f32	%f19, %f18, %f1;\n\
	mul.rn.f32	%f20, %f19, %f19;\n\
	mad.f32	%f21, %f17, %f17, %f20;\n\
	mad.f32	%f22, %f15, %f15, %f21;\n\
	add.rn.f32	%f23, %f22, 0D3FB99999A0000000;\n\
	cvt.f64.f32	%fd0, %f23;\n\
	sqrt.rn.f64	%fd1, %fd0;\n\
	div.rn.f64	%fd3, %fd2, %fd1;\n\
	cvt.rn.f32.f64	%f24, %fd3;\n\
	mul.rn.f32	%f25, %f24, %f24;\n\
	mul.rn.f32	%f26, %f25, %f24;\n\
	ld.global.f32	%f27, [%rd6];\n\
	mul.rn.f32	%f28, %f27, %f26;\n\
	mad.f32	%f8, %f28, %f15, %f3;\n\
	mad.f32	%f7, %f28, %f19, %f4;\n\
	mad.f32	%f6, %f28, %f17, %f5;\n\
	add.u64	%rd10, %rd3, 4;\n\
	add.u64	%rd9, %rd4, 4;\n\
	add.u64	%rd8, %rd5, 4;\n\
	add.u64	%rd7, %rd6, 4;\n\
	add.u32	%r4, %r3, -1;\n\
	setp.ne.u32	%p1, %r4, 0;\n\
	mov.u64	%rd35, %rd10;\n\
	mov.u64	%rd36, %rd9;\n\
	mov.u64	%rd37, %rd8;\n\
	mov.u64	%rd38, %rd7;\n\
	mov.u32	%r10, %r4;\n\
	mov.f32	%f48, %f8;\n\
	mov.f32	%f49, %f7;\n\
	mov.f32	%f50, %f6;\n\
	mov.u64	%rd39, %rd2;\n\
	mov.f32	%f51, %f8;\n\
	mov.f32	%f52, %f7;\n\
	mov.f32	%f53, %f6;\n\
@%p1	bra	$L__BB0_3;\n\
$L__BB0_4:                              // %._crit_edge\n\
	mov.u64	%rd11, %rd39;\n\
	mov.f32	%f9, %f51;\n\
	mov.f32	%f10, %f52;\n\
	mov.f32	%f11, %f53;\n\
	add.u64	%rd28, %rd18, %rd22;\n\
	ld.global.f32	%f29, [%rd28];\n\
	mov.f32	%f30, 0D3FB99999A0000000;\n\
	mad.f32	%f31, %f29, %f30, %f0;\n\
	mov.f32	%f32, 0D3F747AE160000000;\n\
	mad.f32	%f33, %f11, %f32, %f31;\n\
	add.u64	%rd29, %rd15, %rd22;\n\
	st.global.f32	[%rd29], %f33;\n\
	add.u64	%rd30, %rd13, %rd22;\n\
	ld.global.f32	%f34, [%rd30];\n\
	add.u64	%rd31, %rd19, %rd22;\n\
	ld.global.f32	%f35, [%rd31];\n\
	mad.f32	%f36, %f35, %f30, %f34;\n\
	add.u64	%rd32, %rd16, %rd22;\n\
	mad.f32	%f37, %f10, %f32, %f36;\n\
	st.global.f32	[%rd32], %f37;\n\
	add.u64	%rd33, %rd17, %rd22;\n\
	add.u64	%rd34, %rd20, %rd22;\n\
	ld.global.f32	%f38, [%rd11];\n\
	ld.global.f32	%f39, [%rd34];\n\
	mad.f32	%f40, %f39, %f30, %f38;\n\
	mad.f32	%f41, %f9, %f32, %f40;\n\
	st.global.f32	[%rd33], %f41;\n\
	ld.global.f32	%f42, [%rd28];\n\
	mad.f32	%f43, %f11, %f30, %f42;\n\
	st.global.f32	[%rd28], %f43;\n\
	ld.global.f32	%f44, [%rd31];\n\
	mad.f32	%f45, %f10, %f30, %f44;\n\
	st.global.f32	[%rd31], %f45;\n\
	ld.global.f32	%f46, [%rd34];\n\
	mad.f32	%f47, %f9, %f30, %f46;\n\
	st.global.f32	[%rd34], %f47;\n\
	exit;\n\
}\n\
";
var start_time_systemInit = new Date().getTime();

systemInit(); //TODO:  Check if it throws ? // To be called only once, Has CuInit & js-ctypes setup like loading klib & binding functions.

var end_time_systemInit = new Date().getTime();
var systemInit_time = end_time_systemInit - start_time_systemInit;

var start_time_setupKernel = new Date().getTime();
k.setBlockParams(blockSizeX, blockSizeY, 1, blockSizeMultiple); // Pass this in setupKernel ???
k.setGridParams(blockSizeMultiple, blockSizeMultiple);
//for(p =0; p<100; p++){
	var k = setupKernel(kernel_PTX.toString());
//, 3,nProblemSize, 3,nProblemSize, 3,nProblemSize, 3,nProblemSize, 3,nProblemSize, 3,nProblemSize, 3,nProblemSize, 3,nProblemSize,3,nProblemSize,3,nProblemSize
//}

var end_time_setupKernel = new Date().getTime();
var setup_time = end_time_setupKernel - start_time_setupKernel;

var start_time_execute = new Date().getTime();
k.execute(x, y, z, xnew, ynew, znew, vx, vy, vz, m);

var end_time_execute = new Date().getTime();
var execute_time = end_time_execute - start_time_execute;

//alert(start_time_execute);
//alert(end_time_execute);
alert("N-Body : Block sizeX(floats) : " + nProblemSize  + " systemInit : " + systemInit_time + " setup_time : " + setup_time + " execute_time : " + execute_time);


