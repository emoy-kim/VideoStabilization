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