/*
 * Author: Emoy Kim
 * E-mail: emoy.kim_AT_gmail.com
 * 
 * This code is based on 'PatchStabilizer' class among open source in [1], refactored and modified.
 * 
 * [1] https://github.com/LMescheder/Video-Stabilizer/tree/master/libstabilizer
 * 
 */

#pragma once

#include <OpenCVLinker.h>

using namespace std;
using namespace cv;

class PatchStabilization
{
	int PatchColNum;
	int PatchRowNum;
	Mat Homography;
	Mat ReferenceGrayFrame;

	vector<bool> IsValid;
	vector<float> Reliability;
	vector<float> MaxEigenvalues;
	vector<Point2f> CurrentPoints;
	vector<Point2f> ReferencePoints;
	vector<Matx<float, 2, 2>> HarrisMatrices;

	void setReferencePointsAndEigenvalues(const Rect& patch, const int& patch_index, const vector<Mat>& derivatives);
	void initialize(const Mat& reference_gray_frame);

	void updatePointsAndReliability(const Mat& gray_frame);
	void updateHomography(Mat& updated, const Mat& gray_frame);


public:
	PatchStabilization();

	void stabilize(Mat& stabilized, const Mat& frame);
};