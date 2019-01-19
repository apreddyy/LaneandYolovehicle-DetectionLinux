//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//# Please include the Github Repositories web URL if you are using this material.    #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace std;
using namespace cv;
using namespace tensorflow;


extern bool do_calib;



void calibration_on();
void getIndicesOfNonZeroPixels(cuda::GpuMat& src, vector<float>&output_hx, vector<float>&output_hy);
void getIndicesOfNonZeroPixelsnext(cuda::GpuMat& src, vector<float>&Loutput_hx, vector<float>&Loutput_hy, vector<float>&Routput_hx, vector<float>&Routput_hy);

class LANEDETECTION
{
public:
	float center_dist;
	float left_curverad;
	float right_curverad;
	vector<float>polyfiteigen(const std::vector<float> &xv, const std::vector<float> &yv, int order);
	vector<float> polyvaleigen(const std::vector<float>& oCoeff, const std::vector<float>& oX);
	void gray_frame(cuda::GpuMat& src, cuda::GpuMat& dst);
	void binary_frame(cuda::GpuMat& src, cuda::GpuMat& dst);
	void hsv_frame(cuda::GpuMat& src, cuda::GpuMat& dst);
	void wrap_frame(cuda::GpuMat& src, cuda::GpuMat& dst, Point2f* src_points, Point2f* dst_points);
	void sobel_frame(cuda::GpuMat& src, cuda::GpuMat& dst);
	void resize_frame(cuda::GpuMat& src, cuda::GpuMat& dst, int resize_height, int resize_width);
	void erode_dilate(cuda::GpuMat& src, cuda::GpuMat& dst);
	void video_frame(cuda::GpuMat& src, vector<float>&polyleft_out, vector<float>&polyright_out);
	void first_frame(cuda::GpuMat& src, vector<float>&polyright_f, vector<float>&polyleft_f);
	void nxt_frame(cuda::GpuMat& src, vector<float>&polyright_n, vector<float>&polyleft_n);
	void curvature_sanity_check(vector<float>&polyleft_in, vector<float>&polyright_in, vector<int>&Leftx, vector<int>&rightx, vector<int>&main_y);
	void yolodetection(std::vector<Tensor> &outputs, Mat& framein, Mat& frameout);
	void processinga_frame(cuda::GpuMat& src, cuda::GpuMat& resize, cuda::GpuMat& dst, cuda::GpuMat& gpu_mapa, cuda::GpuMat& gpu_mapb);
	void processingb_frame(Mat& frame, cuda::GpuMat& src, cuda::GpuMat& dst);
};

class last_fit
{
public:
	static vector<float>polyright_last;
	static vector<float>polyleft_last;

};










