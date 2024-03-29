/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include<math.h>
extern "C"
void BlackScholesGPU(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{
    //Thread index
    const int      tid = (__builtin_ptx_read_ctaid_x() * __builtin_ptx_read_ntid_x()) + __builtin_ptx_read_tid_x();
    //Total number of threads in execution grid
    const int THREAD_N = __builtin_ptx_read_nctaid_x() * __builtin_ptx_read_ntid_x();

    //No matter how small is execution grid or how large OptN is,
    //exactly OptN indices will be processed with perfect memory coalescing
    for (int opt = tid; opt < optN; opt += THREAD_N) {
	float sqrtT, expRT;
	float d1, d2, CNDD1, CNDD2;

	sqrtT = sqrtf(T);
	d1 = (__logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
	d2 = d1 - V * sqrtT;

	
	

	//CNDD1 = cndGPU(d1);
	const float       A1 = 0.31938153f;
	const float       A2 = -0.356563782f;
	const float       A3 = 1.781477937f;
	const float       A4 = -1.821255978f;
	const float       A5 = 1.330274429f;
	const float RSQRT2PI = 0.39894228040143267793994605993438f;

	float
	K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

	float
	cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
	  (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

	if (d > 0)
		CNDD1 = 1.0f - cnd;


	//CNDD2 = cndGPU(d2);
	K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

	cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
	  (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

	if (d > 0)
		CNDD2 = 1.0f - cnd;

	//Calculate Call and Put simultaneously
	expRT = __expf(- R * T);
	CallResult = S * CNDD1 - X * expRT * CNDD2;
	PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
    }
}
