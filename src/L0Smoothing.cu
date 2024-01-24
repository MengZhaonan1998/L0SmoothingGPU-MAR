#include "header.h"


__global__ void SetSingleValueKernel(cufftReal* d_fpInputData, cufftReal value, size_t nIdx)
{
	int tid = threadIdx.x;
	if (tid == 0)
	{
		d_fpInputData[nIdx] = value;
	}
}


__global__ void ThresholdKernel(cufftReal* d_Array, cufftReal threshold, size_t nSize)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < nSize)
	{
		if (d_Array[tid] < threshold)
			d_Array[tid] = threshold;
	}
}


__global__ void InitVectorKernel(cufftReal* fVector, cufftReal fValue, size_t nSize)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < nSize) {
		fVector[tid] = fValue;
	}
}


__global__ void Denormin2Kernel(cufftReal* d_Denormin2, cufftComplex* d_otfFx, cufftComplex* d_otfFy, size_t nRows, size_t nCols)
{
	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = tx * nCols + ty;
	cufftReal absOtfFx, absOtfFy;
	
	if (tx < nRows && ty < nCols)
	{
		absOtfFx = d_otfFx[idx].x * d_otfFx[idx].x + d_otfFx[idx].y * d_otfFx[idx].y;
		absOtfFy = d_otfFy[idx].x * d_otfFy[idx].x + d_otfFy[idx].y * d_otfFy[idx].y;
		d_Denormin2[idx] = absOtfFx + absOtfFy;
	}
}


__global__ void HSubproblemKernel(cufftReal* d_fpInputData, cufftReal* d_hArray, size_t nRows, size_t nCols)
{
	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = tx * nCols + ty;

	if (tx < nRows && ty < nCols)
	{
		if (ty == (nCols - 1)) {
			d_hArray[idx] = d_fpInputData[tx * nCols] - d_fpInputData[idx];
		}
		else {
			d_hArray[idx] = d_fpInputData[tx * nCols + ty + 1] - d_fpInputData[idx];
		}
	}
}


__global__ void VSubproblemKernel(cufftReal* d_fpInputData, cufftReal* d_vArray, size_t nRows, size_t nCols)
{
	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = tx * nCols + ty;

	if (tx < nRows && ty < nCols)
	{
		if (tx == (nRows - 1)) {
			d_vArray[idx] = d_fpInputData[ty] - d_fpInputData[idx];
		}
		else {
			d_vArray[idx] = d_fpInputData[(tx + 1) * nCols + ty] - d_fpInputData[idx];
		}
	}
}


__global__ void HvValueConditionKernel(cufftReal* d_h, cufftReal* d_v, size_t nSize, cufftReal fLambda, cufftReal fBeta)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	cufftReal temp;
	if (tid < nSize)
	{
		temp = d_h[tid] * d_h[tid] + d_v[tid] * d_v[tid];
		if (temp < fLambda / fBeta)
		{
			d_h[tid] = 0;
			d_v[tid] = 0;
		}
	}
}


__global__ void SSubproblemKernel_h(cufftReal* d_Array, cufftReal* d_h, size_t nRows, size_t nCols)
{
	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = tx * nCols + ty;
	if (tx < nRows && ty < nCols)
	{
		if (ty == 0) {
			d_Array[idx] = d_h[tx * nCols + nCols - 1] - d_h[idx];
		}
		else {
			d_Array[idx] = d_h[tx * nCols + ty - 1] - d_h[idx];
		}
	}
}


__global__ void SSubproblemKernel_v(cufftReal* d_Array, cufftReal* d_v, size_t nRows, size_t nCols)
{
	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = tx * nCols + ty;
	if (tx < nRows && ty < nCols)
	{
		if (tx == 0) {
			d_Array[idx] = d_v[(nRows - 1) * nCols + ty] - d_v[idx];
		}
		else {
			d_Array[idx] = d_v[(tx -1) * nCols + ty] - d_v[idx];
		}
	}
}


__global__ void SSubproblemKernel_FS(cufftComplex* d_FS, cufftComplex* d_Normin1, cufftReal* d_Denormin, cufftReal beta, size_t nSize)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;	
	cufftComplex temp_FS, temp_Normin1;
	cufftReal temp_Denormin;

	if (tid < nSize)
	{
		temp_FS = d_FS[tid];
		temp_Normin1 = d_Normin1[tid];
		temp_Denormin = d_Denormin[tid];

		temp_FS.x = (beta * temp_FS.x + temp_Normin1.x) / temp_Denormin;
		temp_FS.y = (beta * temp_FS.y + temp_Normin1.y) / temp_Denormin;
	
		d_FS[tid] = temp_FS;
	}
}


void L0Smoothing(float* h_fpInputData, size_t nRows, size_t nCols, float fLambda, float fKappa)
{
	auto t_start = std::chrono::high_resolution_clock::now();
	float betamax = 1e5;
	size_t nArraySize = nRows * nCols;
	size_t nFFTArraySize = nRows * (nCols / 2 + 1);
	cufftReal threshold = 0.0;
	cudaError_t cudaStatus;
	cufftHandle cufftPlan1;
	cufftHandle cufftPlan2;
	checkCufftErrors(cufftPlan2d(&cufftPlan1, nRows, nCols, CUFFT_R2C)); // FFT2D
	checkCufftErrors(cufftPlan2d(&cufftPlan2, nRows, nCols, CUFFT_C2R));

	// Copy input data from host to device
	cufftReal* d_fpInputData;
	checkCudaErrors(cudaMalloc((void**)&d_fpInputData, sizeof(cufftReal) * nArraySize));
	checkCudaErrors(cudaMemcpy(d_fpInputData, h_fpInputData, sizeof(cufftReal) * nArraySize, cudaMemcpyHostToDevice));
	
	// Allocation/Initialization d_fx/d_fy
	cufftReal* d_fx, * d_fy;
	checkCudaErrors(cudaMalloc((void**)&d_fx, sizeof(cufftReal) * nArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_fy, sizeof(cufftReal) * nArraySize));
	checkCudaErrors(cudaMemset(d_fx, 0, sizeof(cufftReal) * nArraySize));
	checkCudaErrors(cudaMemset(d_fy, 0, sizeof(cufftReal) * nArraySize));
	
	// Allocation d_otfFx/d_otfFy
	cufftComplex* d_otfFx, * d_otfFy;
	checkCudaErrors(cudaMalloc((void**)&d_otfFx, sizeof(cufftComplex) * nFFTArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_otfFy, sizeof(cufftComplex) * nFFTArraySize));

	// Allocation d_Normin1/d_Normin2/d_Normin2_temp/d_Denormin/d_Denormin2
	cufftComplex* d_Normin1;
	cufftReal* d_Denormin2, * d_Denormin, * d_Normin2, * d_Normin2_temp;
	checkCudaErrors(cudaMalloc((void**)&d_Normin1, sizeof(cufftComplex) * nFFTArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_Normin2, sizeof(cufftReal) * nArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_Normin2_temp, sizeof(cufftReal) * nArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_Denormin, sizeof(cufftReal) * nFFTArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_Denormin2, sizeof(cufftReal) * nFFTArraySize));

	// Allocation d_h/d_v
	cufftReal* d_h, * d_v;
	checkCudaErrors(cudaMalloc((void**)&d_h, sizeof(cufftReal) * nArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_v, sizeof(cufftReal) * nArraySize));

	// Allocation d_FS
	cufftComplex* d_FS;
	checkCudaErrors(cudaMalloc((void**)&d_FS, sizeof(cufftComplex) * nFFTArraySize));

	// Compute the max value of the input data
	int maxIndex = cublasIsamax(nArraySize, d_fpInputData, 1);   // Fairly slow (?)	   //std::sort(h_fpInputData, h_fpInputData + nArraySize);  // 20 ms
	cufftReal maxValue = h_fpInputData[maxIndex];
	cublasSscal(nArraySize, 1.0f / maxValue, d_fpInputData, 1);
	//ScaleVectorKernel<<<(nArraySize + 63) / 64, 64 >>>(d_fpInputData, 1.0f / maxValue, nArraySize);
	
	// PseudoPsf2Otf 
	SetSingleValueKernel << <1, 1 >> > (d_fx, -1, 0); checkCudaErrors(cudaGetLastError());
	SetSingleValueKernel << <1, 1 >> > (d_fx, 1, nCols-1); checkCudaErrors(cudaGetLastError());
	SetSingleValueKernel << <1, 1 >> > (d_fy, -1, 0); checkCudaErrors(cudaGetLastError());
	SetSingleValueKernel << <1, 1 >> > (d_fy, 1, nCols * (nRows - 1)); checkCudaErrors(cudaGetLastError());
	
	checkCufftErrors(cufftExecR2C(cufftPlan1, d_fx, d_otfFx));
	checkCufftErrors(cufftExecR2C(cufftPlan1, d_fy, d_otfFy));
	
	// Compute Normin1: FFT input data
	checkCufftErrors(cufftExecR2C(cufftPlan1, d_fpInputData, d_Normin1));  
	
	// Compute Denomin2
	dim3 threads(16, 16, 1);
	dim3 blocks((nRows + threads.x - 1) / threads.x, (nCols + threads.y - 1) / threads.y, 1);	
	Denormin2Kernel << <blocks, threads >> > (d_Denormin2, d_otfFx, d_otfFy, nRows, nCols / 2 + 1); checkCudaErrors(cudaGetLastError());

	float beta = 2 * fLambda;
	while (beta < betamax)
	{
		// Denorm = 1 + beta * Denormin2
		InitVectorKernel<< < (nFFTArraySize + 63) / 64, 64 >> > (d_Denormin, 1.0, nFFTArraySize); checkCudaErrors(cudaGetLastError());
		cublasSaxpy(nFFTArraySize, beta, d_Denormin2, 1, d_Denormin, 1);

		// h-v subproblem
		HSubproblemKernel << <blocks, threads >> > (d_fpInputData, d_h, nRows, nCols); checkCudaErrors(cudaGetLastError());
		VSubproblemKernel << <blocks, threads >> > (d_fpInputData, d_v, nRows, nCols); checkCudaErrors(cudaGetLastError());

		// h/v condition
		HvValueConditionKernel << <(nArraySize + 63) / 64, 64 >> > (d_h, d_v, nArraySize, fLambda, beta); checkCudaErrors(cudaGetLastError());

		// S subproblem
		SSubproblemKernel_h << < blocks, threads >> > (d_Normin2, d_h, nRows, nCols); checkCudaErrors(cudaGetLastError());	
		SSubproblemKernel_v << < blocks, threads >> > (d_Normin2_temp, d_v, nRows, nCols); checkCudaErrors(cudaGetLastError());	
		cublasSaxpy(nArraySize, 1.0, d_Normin2_temp, 1, d_Normin2, 1);
		checkCufftErrors(cufftExecR2C(cufftPlan1, d_Normin2, d_FS));	
		SSubproblemKernel_FS << <(nFFTArraySize + 63) / 64, 64 >> > (d_FS, d_Normin1, d_Denormin, beta, nFFTArraySize);

		// IFFT
		checkCufftErrors(cufftExecC2R(cufftPlan2, d_FS, d_fpInputData));
		cublasSscal(nArraySize, 1.0f/nArraySize, d_fpInputData, 1);
		
		// threshold
		//ThresholdKernel << <(nArraySize + 63) / 64, 64 >> > (d_fpInputData, threshold, nArraySize);

		beta *= fKappa;
		std::cout << ".";
	}
	
	cublasSscal(nArraySize, maxValue, d_fpInputData, 1);

	checkCudaErrors(cudaMemcpy(h_fpInputData, d_fpInputData, sizeof(cufftReal) * nArraySize, cudaMemcpyDeviceToHost));

	//cufftReal* h_output = new cufftReal[nFFTArraySize];
	//cudaMemcpy(h_output, d_Denormin2, sizeof(cufftReal) * nFFTArraySize, cudaMemcpyDeviceToHost);
	//outputRealMatrix<cufftReal>(h_output, 1, 5);
	//cufftComplex* h_output = new cufftComplex[nFFTArraySize];
	//cudaMemcpy(h_output, d_Normin1, sizeof(cufftComplex) * nFFTArraySize, cudaMemcpyDeviceToHost);
	//outputComplexMatrix(h_output, 1, 5);	

	// Resource relief
	cufftDestroy(cufftPlan1);
	cufftDestroy(cufftPlan2);
	checkCudaErrors(cudaFree(d_fpInputData));
	checkCudaErrors(cudaFree(d_fx));
	checkCudaErrors(cudaFree(d_fy));
	cudaFree(d_otfFx);
	cudaFree(d_otfFy);
	cudaFree(d_Normin1);
	cudaFree(d_Normin2);
	cudaFree(d_Normin2_temp);
	cudaFree(d_Denormin);
	cudaFree(d_Denormin2);
	cudaFree(d_h);
	cudaFree(d_v);
	cudaFree(d_FS);
	
	auto t_end = std::chrono::high_resolution_clock::now();
	auto ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
	std::cout << "L0Smoothing took " << ms_duration.count() << " ms" << std::endl;
	return;
}


void L0SmoothingMultiSlice(float* h_fpInputData, size_t nRows, size_t nCols, size_t nSlices, float fLambda, float fKappa)
{
	auto t_start = std::chrono::high_resolution_clock::now();
	float betamax = 1e5;
	size_t nArraySize = nRows * nCols;
	size_t nFFTArraySize = nRows * (nCols / 2 + 1);
	cudaError_t cudaStatus;
	cufftHandle cufftPlan1;
	cufftHandle cufftPlan2;
	checkCufftErrors(cufftPlan2d(&cufftPlan1, nRows, nCols, CUFFT_R2C)); // FFT2D
	checkCufftErrors(cufftPlan2d(&cufftPlan2, nRows, nCols, CUFFT_C2R)); 

	// Allocation d_fpInputData
	cufftReal* d_fpInputData;
	checkCudaErrors(cudaMalloc((void**)&d_fpInputData, sizeof(cufftReal) * nArraySize));

	// Allocation/Initialization d_fx/d_fy
	cufftReal* d_fx, * d_fy;
	checkCudaErrors(cudaMalloc((void**)&d_fx, sizeof(cufftReal) * nArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_fy, sizeof(cufftReal) * nArraySize));

	// Allocation d_otfFx/d_otfFy
	cufftComplex* d_otfFx, * d_otfFy;
	checkCudaErrors(cudaMalloc((void**)&d_otfFx, sizeof(cufftComplex) * nFFTArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_otfFy, sizeof(cufftComplex) * nFFTArraySize));

	// Allocation d_Normin1/d_Normin2/d_Normin2_temp/d_Denormin/d_Denormin2
	cufftComplex* d_Normin1;
	cufftReal* d_Denormin2, * d_Denormin, * d_Normin2, * d_Normin2_temp;
	checkCudaErrors(cudaMalloc((void**)&d_Normin1, sizeof(cufftComplex) * nFFTArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_Normin2, sizeof(cufftReal) * nArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_Normin2_temp, sizeof(cufftReal) * nArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_Denormin, sizeof(cufftReal) * nFFTArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_Denormin2, sizeof(cufftReal) * nFFTArraySize));

	// Allocation d_h/d_v
	cufftReal* d_h, * d_v;
	checkCudaErrors(cudaMalloc((void**)&d_h, sizeof(cufftReal) * nArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_v, sizeof(cufftReal) * nArraySize));

	// Allocation d_FS
	cufftComplex* d_FS;
	checkCudaErrors(cudaMalloc((void**)&d_FS, sizeof(cufftComplex) * nFFTArraySize));

	for (int i = 0; i < nSlices; i++)
	{
		std::cout << "Slice " << i << ": ";
		auto t_0 = std::chrono::high_resolution_clock::now();
		float betamax = 1e5;

		checkCudaErrors(cudaMemcpy(d_fpInputData, h_fpInputData + i * nCols * nRows, sizeof(cufftReal) * nArraySize, cudaMemcpyHostToDevice));

		// Compute the max value of the input data
		int maxIndex = cublasIsamax(nArraySize, d_fpInputData, 1);   // Fairly slow (?)	   //std::sort(h_fpInputData, h_fpInputData + nArraySize);  // 20 ms
		cufftReal maxValue = h_fpInputData[i * nCols * nRows + maxIndex];
		cublasSscal(nArraySize, 1.0f / maxValue, d_fpInputData, 1);
		//ScaleVectorKernel<<<(nArraySize + 63) / 64, 64 >>>(d_fpInputData, 1.0f / maxValue, nArraySize);

		// PseudoPsf2Otf 
		checkCudaErrors(cudaMemset(d_fx, 0, sizeof(cufftReal) * nArraySize));
		checkCudaErrors(cudaMemset(d_fy, 0, sizeof(cufftReal) * nArraySize));
		SetSingleValueKernel << <1, 1 >> > (d_fx, -1, 0); checkCudaErrors(cudaGetLastError());
		SetSingleValueKernel << <1, 1 >> > (d_fx, 1, nCols - 1); checkCudaErrors(cudaGetLastError());
		SetSingleValueKernel << <1, 1 >> > (d_fy, -1, 0); checkCudaErrors(cudaGetLastError());
		SetSingleValueKernel << <1, 1 >> > (d_fy, 1, nCols * (nRows - 1)); checkCudaErrors(cudaGetLastError());
		
		checkCufftErrors(cufftExecR2C(cufftPlan1, d_fx, d_otfFx));
		checkCufftErrors(cufftExecR2C(cufftPlan1, d_fy, d_otfFy));

		// Compute Normin1: FFT input data
		checkCufftErrors(cufftExecR2C(cufftPlan1, d_fpInputData, d_Normin1));

		// Compute Denomin2
		dim3 threads(16, 16, 1);
		dim3 blocks((nRows + threads.x - 1) / threads.x, (nCols + threads.y - 1) / threads.y, 1);
		Denormin2Kernel << <blocks, threads >> > (d_Denormin2, d_otfFx, d_otfFy, nRows, nCols / 2 + 1); checkCudaErrors(cudaGetLastError());

		float beta = 2 * fLambda;
		while (beta < betamax)
		{
			// Denorm = 1 + beta * Denormin2
			InitVectorKernel<< < (nFFTArraySize + 63) / 64, 64 >> > (d_Denormin, 1.0, nFFTArraySize); checkCudaErrors(cudaGetLastError());
			cublasSaxpy(nFFTArraySize, beta, d_Denormin2, 1, d_Denormin, 1);

			// h-v subproblem
			HSubproblemKernel << <blocks, threads >> > (d_fpInputData, d_h, nRows, nCols); checkCudaErrors(cudaGetLastError());
			VSubproblemKernel << <blocks, threads >> > (d_fpInputData, d_v, nRows, nCols); checkCudaErrors(cudaGetLastError());

			// h/v condition
			HvValueConditionKernel << <(nArraySize + 63) / 64, 64 >> > (d_h, d_v, nArraySize, fLambda, beta); checkCudaErrors(cudaGetLastError());

			// S subproblem
			SSubproblemKernel_h << < blocks, threads >> > (d_Normin2, d_h, nRows, nCols); checkCudaErrors(cudaGetLastError());
			SSubproblemKernel_v << < blocks, threads >> > (d_Normin2_temp, d_v, nRows, nCols); checkCudaErrors(cudaGetLastError());
			cublasSaxpy(nArraySize, 1.0, d_Normin2_temp, 1, d_Normin2, 1);
			checkCufftErrors(cufftExecR2C(cufftPlan1, d_Normin2, d_FS));
			SSubproblemKernel_FS << <(nFFTArraySize + 63) / 64, 64 >> > (d_FS, d_Normin1, d_Denormin, beta, nFFTArraySize); checkCudaErrors(cudaGetLastError());

			// IFFT
			checkCufftErrors(cufftExecC2R(cufftPlan2, d_FS, d_fpInputData));
			cublasSscal(nArraySize, 1.0f / nArraySize, d_fpInputData, 1);

			beta *= fKappa;
			std::cout << ".";	
		}

		cublasSscal(nArraySize, maxValue, d_fpInputData, 1);

		checkCudaErrors(cudaMemcpy(h_fpInputData + i * nCols * nRows, d_fpInputData, sizeof(cufftReal) * nArraySize, cudaMemcpyDeviceToHost));

		auto t_1 = std::chrono::high_resolution_clock::now();
		auto ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_1 - t_0);
		std::cout << ms_duration.count() << " ms" << std::endl;
	}

	// Resource relief
	cufftDestroy(cufftPlan1);
	cufftDestroy(cufftPlan2);
	checkCudaErrors(cudaFree(d_fpInputData));
	checkCudaErrors(cudaFree(d_fx));
	checkCudaErrors(cudaFree(d_fy));
	cudaFree(d_otfFx);
	cudaFree(d_otfFy);
	cudaFree(d_Normin1);
	cudaFree(d_Normin2);
	cudaFree(d_Normin2_temp);
	cudaFree(d_Denormin);
	cudaFree(d_Denormin2);
	cudaFree(d_h);
	cudaFree(d_v);
	cudaFree(d_FS);

	auto t_end = std::chrono::high_resolution_clock::now();
	auto ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
	std::cout << "L0SmoothingMultiSlice took " << ms_duration.count() << " ms" << std::endl;
	return;
}
