#include "header.h"

void SingleSliceTest()
{
	std::cout << "Single-slice test begins!" << std::endl;

	std::string strInputPath = "../TestData\\test_recon_768_768.raw";
	std::string strOutputPath = "../TestData\\output_test_recon_768_768.raw";
	size_t nRows = 768;
	size_t nCols = 768;

	// Load input
	float* h_fpInputData = new float[nRows * nCols];
	ReadRawDataSingle<float>(h_fpInputData, strInputPath.c_str(), nRows * nCols);

	// Run L0Smoothing
	float fLambda = 2e-3;
	float fKappa = 3.0;
	L0Smoothing(h_fpInputData, nRows, nCols, fLambda, fKappa);

	// Save output
	SavetoFile<float>(strOutputPath.c_str(), h_fpInputData, nRows * nCols);
	delete[] h_fpInputData;

	return;
}

void MultiSliceTest()
{
	std::cout << "Multi-slice test begins!" << std::endl;

	std::string strInputPath = "../TestData\\test_recon_768_768_10.raw";
	std::string strOutputPath = "../TestData\\output_test_recon_768_768_10.raw";
	size_t nRows = 768;
	size_t nCols = 768;
	size_t nSlices = 10;

	// Load input
	float* h_fpInputData = new float[nRows * nCols * nSlices];
	ReadRawDataSingle<float>(h_fpInputData, strInputPath.c_str(), nRows * nCols * nSlices);

	// Run L0Smoothing
	float fLambda = 2e-3;
	float fKappa = 3.0;
	L0SmoothingMultiSlice(h_fpInputData, nRows, nCols, nSlices, fLambda, fKappa);

	// Save output
	SavetoFile<float>(strOutputPath.c_str(), h_fpInputData, nRows * nCols * nSlices);
	delete[] h_fpInputData;

	return;
}


int main()
{	
	SingleSliceTest();
	cudaDeviceReset();

	MultiSliceTest();
	cudaDeviceReset();

	return 0;
}
