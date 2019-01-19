//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//# Please include the Github Repositories web URL if you are using this material.    #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "all_header.h"


using namespace std;
using namespace cv;


void LANEDETECTION::erode_dilate(cuda::GpuMat& src, cuda::GpuMat& dst)
{
	cuda::GpuMat erode_out, dilate_out;
	int noise = 3;
	int dilate_const = 1;
	Mat element_erosion = getStructuringElement(MORPH_RECT, Size(noise * 2 + 1, noise * 2 + 1));
	Ptr<cuda::Filter> erode = cuda::createMorphologyFilter(cv::MORPH_ERODE, src.type(), element_erosion);
	erode->apply(src, erode_out);
	Mat element_dilation = getStructuringElement(MORPH_RECT, Size(dilate_const * 2 + 1, dilate_const * 2 + 1));
	Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, src.type(), element_dilation);
	dilateFilter->apply(erode_out, dst);
}