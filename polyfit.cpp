//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//# Please include the Github Repositories web URL if you are using this material.    #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
#include <Eigen/QR>
#include <stdio.h>
#include <vector>
#include <Eigen/Core>
#include "all_header.h"


using namespace std;

//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//# Copyright For polyfiteigen:                                                        //
//  Copyright (C) 2014  RIEGL Research ForschungsGmbH                                  //
//  Copyright(C) 2014  Clifford Wolf <clifford@clifford.at>							   //
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
vector<float>LANEDETECTION::polyfiteigen(const vector<float> &xv, const vector<float> &yv, int order)
{
	Eigen::initParallel();
	Eigen::MatrixXf A = Eigen::MatrixXf::Ones(xv.size(), order + 1);
	Eigen::VectorXf yv_mapped = Eigen::VectorXf::Map(&yv.front(), yv.size());
	Eigen::VectorXf xv_mapped = Eigen::VectorXf::Map(&xv.front(), xv.size());
	Eigen::VectorXf result;

	assert(xv.size() == yv.size());
	assert(xv.size() >= order + 1);

	for (int j = 1; j < order + 1; j++)
	{
		A.col(j) = A.col(j-1).cwiseProduct(xv_mapped);
	}
	
	result = A.householderQr().solve(yv_mapped);
	vector<float> coeff;
	coeff.resize(order + 1);
	for (size_t i = 0; i < order + 1; i++)
		coeff[i] = result[i];

	return coeff;
}
//                       # Copyright For polyfiteigen: END                          //


vector<float>LANEDETECTION::polyvaleigen(const vector<float>& oCoeff,
	const vector<float>& oX)
{
	int nCount = int(oX.size());
	int nDegree = int(oCoeff.size());
	vector<float>oY(nCount);
	
	for (int i = 0; i < nCount; i++)
	{
		float nY = 0;
		float nXT = 1;
		float nX = oX[i];
		for (int j = 0; j < nDegree; j++)
		{
			nY += oCoeff[j] * nXT;
			nXT *= nX;
		}
		oY[i] = nY;
		
	}

	return oY;
}