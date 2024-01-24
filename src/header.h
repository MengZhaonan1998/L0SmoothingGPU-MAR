#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "cublas.h"

void L0Smoothing(float* h_fpInputData, size_t nRows, size_t nCols, float fLambda, float fKappa);
void L0SmoothingMultiSlice(float* h_fpInputData, size_t nRows, size_t nCols, size_t nSlices, float fLambda, float fKappa);

template <class T>
static bool ReadRawDataSingle(T* image, std::string filePath, size_t elementCount)
{
	FILE* fp = fopen(filePath.c_str(), "rb");
	if (fp == NULL)
	{
		std::cout << "Cannot open the file: " << filePath << "!" << std::endl;
		return false;
	}
	size_t readCount;
	readCount = fread(image, sizeof(T), elementCount, fp);
	if (readCount != elementCount)
	{
		std::cout << "The number of elements loaded is smaller than the specified argument!" << std::endl;
		return false;
	}

	fclose(fp);
	return true;
}

template <class T>
static void SavetoFile(std::string filePath, T* data, size_t size)
{
	std::ofstream ofs(filePath, std::ios::out | std::ios::binary);
	ofs.write((const char*)data, sizeof(T) * size);
}

static void checkCudaErrors(cudaError_t cudaError) // -> air_correction
{
	if (cudaError != cudaSuccess)
	{
		std::cout << "cudaError = " << cudaError << ": " << cudaGetErrorName(cudaError) << cudaGetErrorString(cudaError) << std::endl;
	}
}

static void checkCufftErrors(cufftResult cudaError) // -> air_correction
{
	if (cudaError != CUFFT_SUCCESS)
	{
		switch (cudaError)
		{
			case CUFFT_INVALID_PLAN:
				std::cout << "CUFFT_INVALID_PLAN" << std::endl;
			case CUFFT_ALLOC_FAILED:
				std::cout << "CUFFT_ALLOC_FAILED" << std::endl;
			case CUFFT_INVALID_TYPE:
				std::cout << "CUFFT_INVALID_TYPE" << std::endl;
			case CUFFT_INVALID_VALUE:
				std::cout << "CUFFT_INVALID_VALUE" << std::endl;
			case CUFFT_INTERNAL_ERROR:
				std::cout << "CUFFT_INTERNAL_ERROR" << std::endl;
			case CUFFT_EXEC_FAILED:
				std::cout << "CUFFT_EXEC_FAILED" << std::endl;
			case CUFFT_SETUP_FAILED:
				std::cout << "CUFFT_SETUP_FAILED" << std::endl;
			case CUFFT_INVALID_SIZE:
				std::cout << "CUFFT_INVALID_SIZE" << std::endl;
			case CUFFT_UNALIGNED_DATA:
				std::cout << "CUFFT_UNALIGNED_DATA" << std::endl;
		}
	}
}

static void outputComplexMatrix(cufftComplex* complexMatrix, size_t nRows, size_t nCols) 
{
	std::cout << "The complex matrix is:" << "\n";
	for (size_t i = 0; i < nRows; i++)
	{
		for (size_t j = 0; j < nCols; j++)
		{
			std::cout << "(" << complexMatrix[i * nCols + j].x << ", " << complexMatrix[i * nCols + j].y << ") ";
		}
		std::cout << "\n";
	}
}

template <class T>
static void outputRealMatrix(T* realMatrix, size_t nRows, size_t nCols)
{
	std::cout << "The real matrix is:" << "\n";
	for (size_t i = 0; i < nRows; i++)
	{
		for (size_t j = 0; j < nCols; j++)
		{
			std::cout << realMatrix[i * nCols + j] << " ";
		}
		std::cout << "\n";
	}
}
