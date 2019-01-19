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


void LANEDETECTION::hsv_frame(cuda::GpuMat& src, cuda::GpuMat& dst)

{
	cuda::GpuMat hsv_frame, temp;
	cuda::GpuMat channels_device[3];
	cuda::GpuMat channels_device_dest[3];
	cuda::cvtColor(src, hsv_frame, COLOR_BGR2HSV);
	cuda::split(hsv_frame, channels_device);
	cuda::threshold(channels_device[0], channels_device_dest[0],  0, 100, THRESH_BINARY);
	cuda::threshold(channels_device[2], channels_device_dest[1], 210, 255, THRESH_BINARY);
	cuda::threshold(channels_device[2], channels_device_dest[2], 200, 255, THRESH_BINARY);
	cuda::merge(channels_device_dest, 3, temp);
	cuda::cvtColor(temp, dst, COLOR_HSV2BGR);
}