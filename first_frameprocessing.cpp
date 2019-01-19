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


void LANEDETECTION::first_frame(cuda::GpuMat& src, vector<float>&polyright, vector<float>&polyleft)
{

	cuda::GpuMat hist;
	float margin = 60;
	float minpix = 50;
	float windows_no = 8;
	float src_rows = float(src.rows);
	float windows_height = src_rows / windows_no;
	vector<float>main_hx;
	vector<float>main_hy;
	getIndicesOfNonZeroPixels(src, main_hx, main_hy);
	vector<float>leftx;
	vector<float>lefty;
	vector<float>rightx;
	vector<float>righty;
	cuda::reduce(src(Rect(0, src.rows / 2, src.cols, src.rows / 2)), hist, 0, CV_REDUCE_SUM, CV_32S);
	int midpoint = (int(hist.cols / 2));
	Point max_locL, max_locR;
	cuda::minMaxLoc(hist(Rect(50, 0, midpoint, hist.rows)), NULL, NULL, NULL, &max_locL);
	cuda::minMaxLoc(hist(Rect(midpoint, 0, midpoint, hist.rows)), NULL, NULL, NULL, &max_locR);
	float leftxbase = float(int(max_locL.x + 50));
	float rightxbase = float(int(max_locR.x + midpoint));


	for (int window = 1; window <= windows_no; window++)
	{
		vector<float>leftx_t;
		vector<float>lefty_t;
		vector<float>rightx_t;
		vector<float>righty_t;
		float win_y_low = float(int(src_rows - (window + 1)* windows_height));
		float win_y_high = float(int(src_rows - window * windows_height));
		float winxleft_low = float(int(leftxbase - margin));
		float winxleft_high = float(int(leftxbase + margin));
		float winxright_low = float(int(rightxbase - margin));
		float winxright_high = float(int(rightxbase + margin));
		float mean_left = 0;
		float mean_right = 0;

		for (auto idx = 0; idx < main_hy.size(); idx++)
		{
			float good_left_inds = float((float(main_hy[idx]) >= win_y_low) & (float(main_hy[idx])  < win_y_high) & (float(main_hx[idx]) >= winxleft_low) & (float(main_hx[idx])  < winxleft_high));
			float good_right_inds = float((float(main_hy[idx]) >= win_y_low) & (float(main_hy[idx]) < win_y_high) & (float(main_hx[idx]) >= winxright_low) & (float(main_hx[idx]) < winxright_high));
			if (good_left_inds != 0.f)
			{
				leftx_t.push_back(float(main_hx[idx]));
				lefty_t.push_back(float(main_hy[idx]));
				mean_left = mean_left + float(main_hx[idx]);
			}
			if (good_right_inds != 0.f)
			{
				rightx_t.push_back(float(main_hx[idx]));
				righty_t.push_back(float(main_hy[idx]));
				mean_right = mean_right + float(main_hx[idx]);
			}

		}
		if (leftx_t.size() > minpix)
		{
			leftxbase = float(int(mean_left / leftx_t.size()));
		}
		if (rightx_t.size() > minpix)
		{
			rightxbase = float(int(mean_right / rightx_t.size()));
		}

		leftx.insert(leftx.end(), leftx_t.begin(), leftx_t.end());
		lefty.insert(lefty.end(), lefty_t.begin(), lefty_t.end());
		rightx.insert(rightx.end(), rightx_t.begin(), rightx_t.end());
		righty.insert(righty.end(), righty_t.begin(), righty_t.end());


	}

	polyright = LANEDETECTION::polyfiteigen(righty, rightx, 2);
	polyleft = LANEDETECTION::polyfiteigen(lefty, leftx, 2);
}