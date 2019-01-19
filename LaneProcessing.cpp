//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//# Please include the Github Repositories web URL if you are using this material.    #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/types.h"
#include <iostream>
#include <cmath>
#include <numeric>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <functional>
#include <Eigen/Core>
#include "opencv2/opencv.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include "all_header.h"


using namespace std;
using namespace cv;
using namespace tensorflow;


int main(int, const char * const[])
{
	//VideoWriter video("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(640, 360));
	VideoCapture cap("openv.avi");
	if (!cap.isOpened())
	{
		cout << "Error opening video stream or file" << endl;
		return -1;
	}
	Mat mapa, mapb;
	do_calib = false;
	calibration_on();
	Mat intrinsicn = Mat(3, 3, CV_32FC1);
	Mat distCoeffsn = Mat(3, 3, CV_32FC1);;
	cv::FileStorage fs2("calibration.yml", cv::FileStorage::READ);
	fs2["intrinsic"] >> intrinsicn;
	fs2["distCoeffs"] >> distCoeffsn;
	initUndistortRectifyMap(intrinsicn, distCoeffsn, cv::Mat::eye(3, 3, CV_32FC1), intrinsicn, cv::Size(640, 360), CV_32FC1, mapa, mapb);
	while (1)
	{
		Mat frame, cudaout_frame, frame_out, undisort_frame, yolo_frame;
		cuda::GpuMat process_frame, process_frameout, resize_frame, process_framein, gpu_mapa, gpu_mapb, test;

		//# Please include the Github Repositories web URL if you are using this material. Tensorflow    #//
		int depth = 3;
		int height = 448;
		int width = 448;
		

		//# Please include the Github Repositories web URL if you are using this material.  Tensorflow   #//
		Session* session;
		GraphDef graph_def;
		string graph_definition = "graph-vehicledetection.pb";
		std::vector<Tensor>outputs;
		Status status = NewSession(SessionOptions(), &session);
		if (!status.ok())
		{
			std::cout << status.ToString() << "\n";
			return 1;
		}
		TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));
		TF_CHECK_OK(session->Create(graph_def));
		//graph::SetDefaultDevice("/GPU:0", &graph_def);
		std::cout << "GPU assign is done" << std::endl;
		//# Please include the Github Repositories web URL if you are using this material.  Tensorflow   #//

		for (;;)
		{
			cap >> frame;
			if (frame.empty())
				break;
			gpu_mapa.upload(mapa);
			gpu_mapb.upload(mapb);
			process_framein.upload(frame);
			LANEDETECTION lanedetection;
			lanedetection.processinga_frame(process_framein, resize_frame, process_frame, gpu_mapa, gpu_mapb);
			process_frame.download(cudaout_frame);
			
			//# Please include the Github Repositories web URL if you are using this material.  Tensorflow   #//
			cuda::GpuMat resize_frameout, color_frame, normilize_image;
			Mat image;
	
			lanedetection.resize_frame(process_framein, resize_frameout, height, width);
			cuda::cvtColor(resize_frameout, color_frame, cv::COLOR_BGR2RGB);			
			color_frame.convertTo(normilize_image, CV_32FC3, 1.f / 255.f, 0.f);
			normilize_image.download(image);
			const float * source_data = (float*)image.data;
			tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, height, width, depth }));
			auto input_tensor_mapped = input_tensor.tensor<float, 4>();

			for (int y = 0; y < height; ++y)
			{
				const float* source_row = source_data + (y * width * depth);
				for (int x = 0; x < width; ++x)
				{
					const float* source_pixel = source_row + (x * depth);
					for (int c = 0; c < depth; ++c)
					{
						const float* source_value = source_pixel + c;
						input_tensor_mapped(0, y, x, c) = *source_value;
					}
				}
			}
			vector<Tensor> outputTensors;
			session->Run({ { "input" , input_tensor } }, { "output" }, {}, &outputTensors);
			//# Please include the Github Repositories web URL if you are using this material.  Tensorflow   #//

			lanedetection.processingb_frame(cudaout_frame, resize_frame, process_frameout);
			process_frameout.download(frame_out);
			lanedetection.yolodetection(outputTensors, frame_out, yolo_frame);

			//video.write(frame_out);		
			imshow("FrameL", yolo_frame);
			if (waitKey(10) >= 0)break;
		}
		cap.release();
		//video.release();
		destroyAllWindows();
		return 0;
	}
}
