//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//# Please include the Github Repositories web URL if you are using this material.    #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
#include <iostream>
#include <cmath>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
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


void left_point(vector<int>&left_X, vector<int>&main_Y, vector<Point2i>&Pointleft)
{
	int m = int(main_Y.size());
	for (int r = 0; r < m; r++)
	{
		Pointleft.push_back(Point2i(left_X[r], main_Y[r]));
	}

}



void right_point(vector<int>&right_X, vector<int>&main_Y, vector<Point2i>&Pointright)
{

	int m = int(main_Y.size());
	for (int r = 0; r < m; r = r+10)
	{
		
		int c = 359 - r;
		Pointright.push_back(Point2i(right_X[c], main_Y[c]));

	}

}



void LANEDETECTION::processingb_frame(Mat& frame, cuda::GpuMat& src, cuda::GpuMat& dst)
{
	cuda::GpuMat unwrap_framein, unwrap_frameout, cuda_frameout,dilate_out;
	cv::Point2f src_points[4];
	cv::Point2f dst_points[4];

	src_points[0] = cv::Point2f(290, 230);
	src_points[1] = cv::Point2f(350, 230);
	src_points[2] = cv::Point2f(520, 340);
	src_points[3] = cv::Point2f(130, 340);

	dst_points[0] = cv::Point2f(130, 0);
	dst_points[1] = cv::Point2f(520, 0);
	dst_points[2] = cv::Point2f(520, 360);
	dst_points[3] = cv::Point2f(130, 360);
	
	vector<Point2i> nonZeroCoordinates;
	vector<float>polyleft_in;
	vector<float>polyright_in;
	vector<int>Leftx;
	vector<int>rightx;
	vector<int>main_y;

	cuda_frameout.upload(frame);
	LANEDETECTION::erode_dilate(cuda_frameout, dilate_out);
	LANEDETECTION::video_frame(dilate_out, polyleft_in, polyright_in);
	LANEDETECTION::curvature_sanity_check(polyleft_in, polyright_in, Leftx, rightx, main_y);
	
	Mat maskImage = Mat(frame.size(), CV_8UC3, Scalar(0));
	vector<Point2i>PointLeft;
	vector<Point2i>PointRight;

	left_point(Leftx, main_y, PointLeft);
	
	right_point(rightx, main_y, PointRight);

	vector<Point2i>PointLeftRight;
	PointLeft.insert(PointLeft.end(), PointRight.begin(), PointRight.end());
	PointLeftRight = PointLeft;

	polylines(maskImage, PointLeft, false, Scalar(0, 0, 255), 20, 150, 0);
	polylines(maskImage, PointRight, false, Scalar(0, 0, 255), 20, 150, 0);

	const Point *pts = (const cv::Point*)Mat(PointLeftRight).data;
	int npts = Mat(PointLeftRight).rows;
	fillPoly(maskImage, &pts, &npts, 1, Scalar(0, 255, 0), 8);
	unwrap_framein.upload(maskImage);
	LANEDETECTION::wrap_frame(unwrap_framein, unwrap_frameout, dst_points, src_points);

	cuda::addWeighted(src, 1, unwrap_frameout, 0.5, -1, dst);
}

