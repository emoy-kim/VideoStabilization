#include "PatchStabilization.h"

PatchStabilization::PatchStabilization() : PatchColNum( 20 ), PatchRowNum( 15 )
{
	Homography = cv::Mat::eye(3, 3, CV_32FC1);
}

void PatchStabilization::setReferencePointsAndEigenvalues(
	const cv::Rect& patch, 
	int patch_index, 
	const std::vector<cv::Mat>& derivatives
)
{
	const cv::Mat& IxIx = derivatives[0];
	const cv::Mat& IxIy = derivatives[1];
	const cv::Mat& IyIy = derivatives[2];

	float max_trace = -1.0f;
	for (int j = patch.y; j < patch.br().y; ++j) {
		for (int i = patch.x; i < patch.br().x; ++i) {
			const cv::Matx<float, 2, 2> harris_matrix = { 
				IxIx.at<float>(j, i), IxIy.at<float>(j, i),
				IxIy.at<float>(j, i), IyIy.at<float>(j, i)
			};

			const float trace = harris_matrix(0, 0) + harris_matrix(1, 1);
			if (trace > max_trace) {
				HarrisMatrices[patch_index] = harris_matrix;
				ReferencePoints[patch_index] = cv::Point2f(static_cast<float>(i), static_cast<float>(j));

				max_trace = trace;
				const float t = 
					(harris_matrix(0, 0) - harris_matrix(1, 1)) * (harris_matrix(0, 0) - harris_matrix(1, 1)) + 
					4.0f * harris_matrix(0, 1) * harris_matrix(1, 0);
				MaxEigenvalues[patch_index] = 0.5f * (trace + std::sqrt( t ));
			}
		}
	}
}

void PatchStabilization::initialize(const cv::Mat& reference_gray_frame)
{
	ReferenceGrayFrame = reference_gray_frame.clone();

	cv::Mat blurred;
	cv::GaussianBlur( ReferenceGrayFrame, blurred, cv::Size(5, 5), 1.0 );
	blurred.convertTo( blurred, CV_32FC1 );

	cv::Mat dx, dy;
	cv::Sobel( blurred, dx, CV_32FC1, 1, 0 );
	cv::Sobel( blurred, dy, CV_32FC1, 0, 1 );

	std::vector<cv::Mat> derivatives(3);
	cv::multiply( dx, dx, derivatives[0] );
	cv::multiply( dx, dy, derivatives[1] );
	cv::multiply( dy, dy, derivatives[2] );

	cv::GaussianBlur( derivatives[0], derivatives[0], cv::Size(5, 5), 2.0 );
	cv::GaussianBlur( derivatives[1], derivatives[1], cv::Size(5, 5), 2.0 );
	cv::GaussianBlur( derivatives[2], derivatives[2], cv::Size(5, 5), 2.0 );

	HarrisMatrices.resize( PatchColNum * PatchRowNum );
	ReferencePoints.resize( PatchColNum * PatchRowNum );
	MaxEigenvalues.resize( PatchColNum * PatchRowNum );

	const auto patch_width = static_cast<int>(ReferenceGrayFrame.cols) / PatchColNum;
	const auto patch_height = static_cast<int>(ReferenceGrayFrame.rows) / PatchRowNum;
	for (int pj = 0; pj < PatchRowNum; ++pj) {
		for (int pi = 0; pi < PatchColNum; ++pi) {
			const int patch_index = pj * PatchColNum + pi;
			const cv::Rect patch(static_cast<int>(pi * patch_width), static_cast<int>(pj * patch_height), patch_width, patch_height);
			setReferencePointsAndEigenvalues( patch, patch_index, derivatives );
		}
	}

	IsValid.resize( ReferencePoints.size(), true );
	Reliability.resize( ReferencePoints.size(), 1.0 );
}

void PatchStabilization::updatePointsAndReliability(const cv::Mat& gray_frame)
{
	static const int level = 3;
	static const float min_eigen_threshold = 1e-6f;
	static const cv::Size window_size(21, 21);
	std::vector<float> errors;

	std::vector<cv::Point2f> target_points;
	std::vector<uchar> forward_found_matches;
	cv::calcOpticalFlowPyrLK( 
		ReferenceGrayFrame, 
		gray_frame, 
		ReferencePoints, 
		target_points, 
		forward_found_matches, 
		errors, 
		window_size, 
		level,
		cv::TermCriteria(), 0, min_eigen_threshold
	);

	std::vector<cv::Point2f> re_reference_points;
	std::vector<uchar> backward_found_matches;
	cv::calcOpticalFlowPyrLK( 
		gray_frame, 
		ReferenceGrayFrame, 
		target_points, 
		re_reference_points, 
		backward_found_matches, 
		errors, 
		window_size, 
		level,
		cv::TermCriteria(), 0, min_eigen_threshold
	);

	CurrentPoints = target_points;
	for (size_t i = 0; i < CurrentPoints.size(); ++i) {
		const float reproject_error[2] = {
			ReferencePoints[i].x - re_reference_points[i].x, 
			ReferencePoints[i].y - re_reference_points[i].y 
		};
		const float weighted_error = (
			HarrisMatrices[i](0, 0) * reproject_error[0] * reproject_error[0] +
			(HarrisMatrices[i](0, 1) + HarrisMatrices[i](1, 0)) * reproject_error[0] * reproject_error[1] +
			HarrisMatrices[i](1, 1) * reproject_error[1] * reproject_error[1]
		) / MaxEigenvalues[i];

		IsValid[i] = 
			forward_found_matches[i] && backward_found_matches[i] && weighted_error < 1e-1f &&
			reproject_error[0] * reproject_error[0] + reproject_error[1] * reproject_error[1] < 1E+3f  ;
		Reliability[i] = 0.95f * Reliability[i];
		if (IsValid[i]) Reliability[i] += 0.05f;
	}
}

void PatchStabilization::updateHomography(cv::Mat& updated, const cv::Mat& gray_frame)
{
	updatePointsAndReliability( gray_frame );

	cv::Matx<float, 3, 3> estimated_homography = cv::Matx<float, 3, 3>::eye();
	cv::Matx<float, 8, 1> h = cv::Matx<float, 8, 1>::zeros();

	std::vector<cv::Point2f> reprojected_points;
	for (uint iter = 0; iter < 50; ++iter) {
		cv::perspectiveTransform( CurrentPoints, reprojected_points, estimated_homography.inv() );
		cv::Matx<float, 8, 8> A = cv::Matx<float, 8, 8>::zeros();
		cv::Matx<float, 8, 1> b = cv::Matx<float, 8, 1>::zeros();

		float sum_weights = 0.0f;
		for (uint i = 0; i < ReferencePoints.size(); ++i) {
			if (IsValid[i] && Reliability[i] >= 0.5) {
				cv::Matx<float, 2, 1> p0 = { ReferencePoints[i].x, ReferencePoints[i].y };
				cv::Matx<float, 2, 1> p1 = { CurrentPoints[i].x, CurrentPoints[i].y };
				cv::Matx<float, 2, 1> reproject_error = { 
					reprojected_points[i].x - ReferencePoints[i].x, 
					reprojected_points[i].y - ReferencePoints[i].y 
				};
		 
				const float weight = iter == 0 ? 1.0f : 
					1.0f / (1.0f + sqrt( reproject_error.dot( HarrisMatrices[i] * reproject_error ) )) / (h(6) * p0(0)+ h(7) * p0(1) + 1.0f);
				const cv::Matx<float, 2, 8> J = {
					p0(0), p0(1), 1.0f, 0.0f, 0.0f, 0.0f, -p1(0) * p0(0), -p1(0) * p0(1),
					0.0f, 0.0f, 0.0f, p0(0), p0(1), 1.0f, -p1(1) * p0(0), -p1(1) * p0(1)
				};
				b += weight * J.t() * HarrisMatrices[i] * (p1 - p0);
				A += weight * J.t() * HarrisMatrices[i] * J;
				sum_weights += weight;
			}
		}

		A *= 1.0f / sum_weights;
		b *= 1.0f / sum_weights;
		cv::solve( A, b, h, cv::DECOMP_CHOLESKY );

		estimated_homography = {
			1.0f + h(0), h(1), h(2),
			h(3), 1.0f + h(4), h(5),
			h(6), h(7), 1.0f 
		};
	}

	updated = cv::Mat(estimated_homography.inv());
}

void PatchStabilization::stabilize(cv::Mat& stabilized, const cv::Mat& frame)
{
	cv::Mat gray_frame;
	cv::cvtColor( frame, gray_frame, cv::COLOR_BGR2GRAY );

	if (ReferenceGrayFrame.empty()) initialize( gray_frame );

	cv::warpPerspective( gray_frame, gray_frame, Homography, gray_frame.size() );

	cv::Mat updated_homography;
	updateHomography( updated_homography, gray_frame );

	if (updated_homography.empty()) {
		Homography = cv::Mat::eye(3, 3, CV_32FC1);
		stabilized = frame.clone();
		return;
	}

	Homography = updated_homography * Homography;

	cv::warpPerspective( frame, stabilized, Homography, frame.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT );
}