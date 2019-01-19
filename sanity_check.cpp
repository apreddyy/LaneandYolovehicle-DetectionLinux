//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//# Please include the Github Repositories web URL if you are using this material.    #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
#include <stdio.h>
#include <numeric>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "all_header.h"


using namespace std;


float center_point(float x1, float x2)
{
	return  (x1 + x2) / 2.f;
}

float distance(float x1, float x2)
{
	return ((sqrt(pow(float(x1 - x2), 2.f))));
}

vector<float>LinearSpacedArray(float a, float b, size_t N)
{
	float h = (b - a) / static_cast<double>(N - 1);
	std::vector<float> xs(N);
	std::vector<float>::iterator x;
	float val;
	for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
		*x = val;
	}
	return xs;
}

void LANEDETECTION::curvature_sanity_check(vector<float>&polyleft_in, vector<float>&polyright_in, vector<int>&Leftx, vector<int>&rightx, vector<int>&main_y)
{
	float xm_per_pix = 3.7f / 350.0f;
	float ym_per_pix = 30.0f / 360.0f;
	vector<float> Plot_ys(360);
	iota(begin(Plot_ys), end(Plot_ys), 0.f);

	vector<float>Leftx_out_x;
	vector<float>rightx_out_x;
	vector<float> Plot_y = LinearSpacedArray(0.f, 20.f, 10);
	Leftx_out_x = LANEDETECTION::polyvaleigen(polyleft_in, Plot_y);
	rightx_out_x = LANEDETECTION::polyvaleigen(polyright_in, Plot_y);
	float Lmean = float(accumulate(Leftx_out_x.begin(), Leftx_out_x.end(), 0.0) / Leftx_out_x.size());
	float Rmean = float(accumulate(rightx_out_x.begin(), rightx_out_x.end(), 0.0) / rightx_out_x.size());
	float delta_lines = (Rmean - Lmean);

	float L_0 = 2 * polyleft_in[2] * 180 + polyleft_in[1];
	float R_0 = 2 * polyright_in[2] * 180 + polyright_in[1];
	float delta_slope_mid = abs(R_0 - L_0);

	float L_1 = 2 * polyleft_in[2] * 360 + polyleft_in[1];
	float R_1 = 2 * polyright_in[2] * 360 + polyright_in[1];
	float delta_slope_bottom = abs(L_1 - R_1);

	float L_2 = 2 * polyleft_in[2] + polyleft_in[1];
	float R_2 = 2 * polyright_in[2] + polyright_in[1];
	float delta_slope_top = abs(L_2 - R_2);

	vector<float>Leftx_sanity;
	vector<float>rightx_sanity;

	if (((delta_slope_top <= 0.9) && (delta_slope_bottom <= 0.9) && (delta_slope_mid <= 0.9)) && ((delta_lines > 75)))
	{
		last_fit::polyleft_last = polyleft_in;
		last_fit::polyright_last = polyright_in;
		Leftx_sanity = polyleft_in;
		rightx_sanity = polyright_in;
	}
	else
	{
		Leftx_sanity = last_fit::polyleft_last;
		rightx_sanity = last_fit::polyright_last;
	}

	vector<float>Leftx_out;
	vector<float>rightx_out;

	LANEDETECTION polyfitpolyval;
	
	Leftx_out = polyfitpolyval.polyvaleigen(Leftx_sanity, Plot_ys);
	
	rightx_out = polyfitpolyval.polyvaleigen(rightx_sanity, Plot_ys);

	vector<float>Leftx_out_m = Leftx_out;
	vector<float>rightx_out_m = rightx_out;
	vector<float>Plot_ysm = Plot_ys;
	float first_element_L = Leftx_out[359];
	float first_element_R = rightx_out[359];

	float center_x = center_point(first_element_L, first_element_R);
	float center_ix = 320;
	LANEDETECTION::center_dist = (distance(center_x, center_ix)*xm_per_pix);
	//cout << center_dist <<endl;
	
	transform(Leftx_out.begin(), Leftx_out.end(), Leftx_out.begin(),
		bind1st(std::multiplies<float>(), xm_per_pix));



	transform(rightx_out.begin(), rightx_out.end(), rightx_out.begin(),
		bind1st(std::multiplies<float>(), xm_per_pix));



	transform(Plot_ys.begin(), Plot_ys.end(), Plot_ys.begin(),
		bind1st(std::multiplies<float>(), ym_per_pix));



	vector<float>left_fit_cr;
	vector<float>right_fit_cr;

	
	left_fit_cr = polyfitpolyval.polyfiteigen(Plot_ys, Leftx_out, 2);
	LANEDETECTION::left_curverad = float((1 + pow(pow((2 * left_fit_cr[2] * 359 * ym_per_pix + left_fit_cr[1]), 2), 1.5)) / abs(2 * left_fit_cr[2]));
	
	
	right_fit_cr = polyfitpolyval.polyfiteigen(Plot_ys, rightx_out, 2);
	LANEDETECTION::right_curverad = float((1 + pow(pow((2 * right_fit_cr[2] * 359 * ym_per_pix + right_fit_cr[1]), 2), 1.5)) / abs(2 * right_fit_cr[2]));


	rightx.insert(rightx.end(), rightx_out_m.begin(), rightx_out_m.end());
	Leftx.insert(Leftx.end(), Leftx_out_m.begin(), Leftx_out_m.end());
	main_y.insert(main_y.end(), Plot_ysm.begin(), Plot_ysm.end());

}
