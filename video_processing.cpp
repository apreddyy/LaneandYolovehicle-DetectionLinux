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


vector<float>last_fit::polyleft_last;
vector<float>last_fit::polyright_last;


void LANEDETECTION::video_frame(cuda::GpuMat& src, vector<float>&polyleft_out, vector<float>&polyright_out)
{


	if ((last_fit::polyleft_last.size() == 0) && (0 == last_fit::polyright_last.size()))
	{
		LANEDETECTION::first_frame(src, polyright_out, polyleft_out);
		last_fit::polyright_last = polyright_out;
		last_fit::polyleft_last = polyleft_out;
	}
	else
	{
		LANEDETECTION::nxt_frame(src, polyright_out, polyleft_out);
	}

}