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

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using uchar = unsigned char;
using uint = unsigned int;

class PatchStabilization
{
public:
	PatchStabilization();
	~PatchStabilization() = default;

	void stabilize(cv::Mat& stabilized, const cv::Mat& frame);

private:
	int PatchColNum;
	int PatchRowNum;
	cv::Mat Homography;
	cv::Mat ReferenceGrayFrame;

	std::vector<bool> IsValid;
	std::vector<float> Reliability;
	std::vector<float> MaxEigenvalues;
	std::vector<cv::Point2f> CurrentPoints;
	std::vector<cv::Point2f> ReferencePoints;
	std::vector<cv::Matx<float, 2, 2>> HarrisMatrices;

	void setReferencePointsAndEigenvalues(const cv::Rect& patch, int patch_index, const std::vector<cv::Mat>& derivatives);
	void initialize(const cv::Mat& reference_gray_frame);

	void updatePointsAndReliability(const cv::Mat& gray_frame);
	void updateHomography(cv::Mat& updated, const cv::Mat& gray_frame);
};