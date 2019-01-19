//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//# Please include the Github Repositories web URL if you are using this material.    #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
#include <iostream>
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


void LANEDETECTION::resize_frame(cuda::GpuMat& src, cuda::GpuMat& dst, int resize_height, int resize_width)
{
	cuda::resize(src, dst, Size(resize_width, resize_height));
}