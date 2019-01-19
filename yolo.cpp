//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//# Please include the Github Repositories web URL if you are using this material.    #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include <cmath>
#include <numeric>
#include <stddef.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <Eigen/Core>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include "all_header.h"


using namespace tensorflow;
using namespace std;
using namespace cv;


float overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2.;
	float l2 = x2 - w2 / 2.;
	float leftx = max(l1, l2);
	float r1 = x1 + w1 / 2.;
	float r2 = x2 + w2 / 2.;
	float rightx = min(r1, r2);
	return rightx - leftx;
}

void LANEDETECTION::yolodetection(std::vector<Tensor> &outputs, Mat& framein, Mat& frameout)
{

	int Width = 640;
	int Height = 360;
	int S = 7;
	int B = 2;
	int C = 5;
	int SS = S * S;
	int prob_size = SS * C;
	int conf_size = SS * B;
	int seq = prob_size + conf_size;
	float main_threshold = 0.3f;
	float class_threshold = 0.3f;
	float box_threshold = 0.4f;


	tensorflow::Tensor out = std::move(outputs.at(0));
	auto out_put = out.flat<float>();
	vector<float>net_out(735);

	for (int idx0 = 0; idx0 < 735; idx0++)
	{
		net_out[idx0] = (float)(out_put(idx0));
	}
	vector<float>probs = std::vector<float>(net_out.begin(), net_out.begin() + prob_size);
	vector<float>confs = std::vector<float>(net_out.begin() + prob_size, net_out.begin() + seq);
	vector<float>cords = std::vector<float>(net_out.begin() + seq, net_out.end());

	float cords_converted[2][4][49];
	float confs_converted[49][2];
	float probs_converted[49][5];
	int idx = 0, idx1 = 0, idx2 = 0;
	for (int i = 0; i < SS; ++i)
	{
		for (int j = 0; j < B; ++j)
		{
			confs_converted[i][j] = confs[idx1];
			idx1 = idx1 + 1;
			for (int k = 0; k < 4; ++k)
			{
				cords_converted[j][k][i] = cords[idx];
				idx = idx + 1;
			}
		}
		for (int l = 0; l < C; ++l)
		{
			probs_converted[i][l] = probs[idx2];
			idx2 = idx2 + 1;
		}
	}
	vector<float>classes(98);
	vector<float>x_cord(98);
	vector<float>y_cord(98);
	vector<float>width_p(98);
	vector<float>height_p(98);
	vector<float>max_probility(98);
	vector<float>max_index(98);
	vector<vector<float> > problity_class(98, vector<float>(5));
	vector<float>temp(5);
	int id = 0;
	for (int i = 0; i < SS; ++i)
	{
		for (int j = 0; j < B; ++j)
		{
			classes[id] = confs_converted[i][j];
			x_cord[id] = (cords_converted[j][0][i] + (i % S)) / S;
			y_cord[id] = (cords_converted[j][1][i] + float(floor(i / S))) / S;
			width_p[id] = pow(cords_converted[j][2][i], 2);
			height_p[id] = pow(cords_converted[j][3][i], 2);
			for (int k = 0; k < 5; ++k)
			{
				problity_class[id][k] = probs_converted[i][k] * confs_converted[i][j];
				if (pow(problity_class[id][k], 2) < main_threshold)
				{
					problity_class[id][k] = 0.f;
				}
				temp[k] = problity_class[id][k];
			}
			max_probility[id] = *max_element(temp.begin(), temp.end());
			max_index[id] = distance(temp.begin(), max_element(temp.begin(), temp.end()));
			id = id + 1;
		}
	}
	float area;
	for (int i = 0; i < 2 * SS; i++)
	{
		for (int j = i + 1; j < 2 * SS; j++)
		{
			float x1 = x_cord[i];
			float x2 = x_cord[j];
			float y1 = y_cord[i];
			float y2 = y_cord[j];
			float w1 = width_p[i];
			float w2 = width_p[j];
			float h1 = height_p[i];
			float h2 = height_p[j];
			float W = overlap(x1, w1, x2, w2);
			float H = overlap(y1, h1, y2, h2);
			if (W < 0.f || H < 0.f)
			{
				area = 0.f;
			}
			else
			{
				area = W*H;
			}

			float box_iou = area / ((w1* h1) + (w2 * h2) - area);
			if (box_iou >= box_threshold)
			{
				max_probility[j] = 0.0;
			}
		}

	}

	vector<Rect>Rect_points;
	vector<int>classes_detected;
	for (int idx0 = 0; idx0 < 2 * SS; idx0++)
	{
		if (max_probility[idx0] > class_threshold)
		{
			int left = int((x_cord[idx0] - width_p[idx0] / 2.f)*Width);
			int right = int((x_cord[idx0] + width_p[idx0] / 2.f)*Width);
			int top = int((y_cord[idx0] - height_p[idx0] / 2.f)*Height);
			int bot = int((y_cord[idx0] + height_p[idx0] / 2.f)*Height);

			if (left < 0)
			{
				left = 0;
			}
			if (right > Width - 1)
			{
				right = Width - 1;
			}
			if (top < 0)
			{
				top = 0;
			}
			if (bot > Height - 1)
			{
				bot = Height - 1;
			}

			Point2i Pt1 = Point2i(left, top);
			Point2i Pt2 = Point2i(right, bot);
			Point2i Pt3 = Point2i(left, top - 10);
			Rect R(Pt1, Pt2);

			int classes_d = max_index[idx0];
			switch (classes_d)
			{
			case 0:
				putText(framein, "CAR", Pt3, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(255, 255, 255));
				rectangle(framein, R, Scalar(255, 255, 255), 5);
				break;
			case 1:
				putText(framein, "CYCLIST", Pt3, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(255, 0, 0));
				rectangle(framein, R, Scalar(255, 0, 0), 5);
				break;
			case 2:
				putText(framein, "TRAFFICLIGHTS", Pt3, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 255, 0));
				rectangle(framein, R, Scalar(0, 255, 0), 5);
				break;
			case 3:
				putText(framein, "TRUCK", Pt3, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 0, 255));
				rectangle(framein, R, Scalar(0, 0, 255), 5);
				break;
			case 4:
				putText(framein, "PEDSTRIAN", Pt3, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(0, 0, 0));
				rectangle(framein, R, Scalar(0, 0, 0), 5);
				break;
			}
		}
	}


	frameout = framein;

}
