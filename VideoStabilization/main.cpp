#include "PatchStabilization.h"
#include <chrono>

using namespace chrono;

void getTestset(vector<string>& testset)
{
	testset = {
		"VideoSamples/test1.avi",
		"VideoSamples/test2.avi",
		"VideoSamples/test3.avi"
	};
}

#define PAUSE ' '
int displayStabilizedFrame(const Mat& frame, const Mat& stabilized, const bool& to_pause, const bool& screen_merged = false)
{
	if (screen_merged) {
		const Mat result = Mat::zeros(frame.rows, frame.cols * 2, CV_8UC3);
		const int col_diff_size = (frame.cols - stabilized.cols) / 2;
		const int row_diff_size = (frame.rows - stabilized.rows) / 2;
		frame.copyTo( result(Rect(0, 0, frame.cols, frame.rows)) );
		stabilized.copyTo( result(Rect(frame.cols + col_diff_size, row_diff_size, stabilized.cols, stabilized.rows)) );
		imshow( "Input | Heat-Haze Removed", result );
	}
	else {
		imshow( "Input", frame );
		imshow( "Stabilized", stabilized );
	}

	const int key = waitKey( 1 );
	if (to_pause) return PAUSE;
	return key;
}

#define ESC 27
#define TO_BE_CLOSED true
#define TO_BE_CONTINUED false
bool processKeyPressed(bool& to_pause, const int& key_pressed)
{
	switch (key_pressed) {
	case PAUSE: {
		int key;
		while ((key = waitKey( 1 )) != PAUSE && key != 'f') {}
		to_pause = key == 'f';
	} break;
	case ESC:
		return TO_BE_CLOSED;
	default:
		break;
	}
	return TO_BE_CONTINUED;
}

void playVideoAndStabilize(VideoCapture& cam, PatchStabilization& stabilizer)
{
	int key_pressed = -1;
	bool to_pause = false;
	Mat frame, stabilized;
	while (true) {
		cam >> frame;
		if (frame.empty()) break;

		time_point<system_clock> start = system_clock::now();
		stabilizer.stabilize( stabilized, frame );
		const duration<double> stabilization_process_time = (system_clock::now() - start) * 1000.0;
		cout << "PROCESS TIME: " << stabilization_process_time.count() << " ms... \r";

		key_pressed = displayStabilizedFrame( frame, stabilized, to_pause, true );
		if (processKeyPressed( to_pause, key_pressed ) == TO_BE_CLOSED) break;
	}
}

void runTestSet(const vector<string>& testset)
{
	VideoCapture cam;
	for (auto const &test_data : testset) {
		cam.open( test_data );
		if (!cam.isOpened()) continue;

		const int width = static_cast<int>(cam.get( CV_CAP_PROP_FRAME_WIDTH ));
		const int height = static_cast<int>(cam.get( CV_CAP_PROP_FRAME_HEIGHT ));
		cout << "*** TEST SET(" << width << " x " << height << "): " << test_data.c_str() << "***" << endl;

		PatchStabilization stabilizer;
		playVideoAndStabilize( cam, stabilizer );
		cam.release();
	}
}

int main()
{
	vector<string> testset;
	getTestset( testset );
	runTestSet( testset );

	return 0;
}