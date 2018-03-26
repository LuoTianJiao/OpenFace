///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltru歛itis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru歛itis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltru歛itis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltru歛itis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////
// FaceTrackingVid.cpp : Defines the entry point for the console application for tracking faces in videos.

// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"

#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}


int main (int argc, char **argv)
{

	vector<string> arguments = get_arguments(argc, argv);

	LandmarkDetector::FaceModelParameters det_parameters(arguments);

	// The modules that are being used for tracking
	LandmarkDetector::CLNF face_model(det_parameters.model_location);

	// Open a sequence
	Utilities::SequenceCapture sequence_reader;

	// A utility for visualizing the results (show just the tracks)
	Utilities::Visualizer visualizer(true, false, false);
	

	// Tracking FPS for visualization
	Utilities::FpsTracker fps_tracker;
	fps_tracker.AddFrame();

	int sequence_number = 0;
	fstream fileleft, fileright,fileface;
	fileleft.open("lefteyeliklyhood.txt");
	fileright.open("righteyeLikelyHood.txt");
	fileface.open("facelikelyhood.txt");
	while (true) // this is not a for loop as we might also be reading from a webcam
	{

		// The sequence reader chooses what to open based on command line arguments provided
		if(!sequence_reader.Open(arguments))
		{
			// If failed to open because no input files specified, attempt to open a webcam
			if (sequence_reader.no_input_specified && sequence_number == 0)
			{
				// If that fails, revert to webcam
				INFO_STREAM("No input specified, attempting to open a webcam 0");
				if (!sequence_reader.OpenWebcam(0))
				{
					ERROR_STREAM("Failed to open the webcam");
					break;
				}
			}
			else
			{
				ERROR_STREAM("Failed to open a sequence");
				break;
			}
		}
		INFO_STREAM("Device or file opened");

		cv::Mat captured_image = sequence_reader.GetNextFrame();

		INFO_STREAM("Starting tracking");
		while (!captured_image.empty()) // this is not a for loop as we might also be reading from a webcam
		{

			// Reading the images
			cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame();//GetNextFrame()函数中已经生成灰度图，这个函数只是把灰度图调了出来
			DWORD dwBegin = GetTickCount();
			// The actual facial landmark detection / tracking
			bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_model, det_parameters);
			DWORD dwEnd = GetTickCount();
			double timedif = dwEnd - dwBegin;
		///	printf("landmark_time_cost:%f \n", timedif);
			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);

			// If tracking succeeded and we have an eye model, estimate gaze
			if (detection_success && face_model.eye_model)
			{
				GazeAnalysis::EstimateGaze(face_model, gazeDirection0, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
				GazeAnalysis::EstimateGaze(face_model, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);
			}

			// Work out the pose of the head from the tracked model
			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
			// Keeping track of FPS
			fps_tracker.AddFrame(); //记录一个时间，以供下一次判断是否在跟踪时间内
			double eyedataleft = face_model.hierarchical_models[1].model_likelihood;
			double eyedataright = face_model.hierarchical_models[2].model_likelihood;
			double facedata = face_model.model_likelihood;
			fileleft << eyedataleft << endl;
			fileright << eyedataright << endl;
			fileface << facedata << endl;
			// Displaying the tracking visualizations
			if ((pose_estimate[4]<0.5&& pose_estimate[4]>-0.5)&& (pose_estimate[3]<0.5&& pose_estimate[3]>-0.5)
				&& (pose_estimate[5]<0.5&& pose_estimate[5]>-0.5))
			{
				if (eyedataleft < (-2.0) && eyedataright < (-2.0))
				{
					static int i = 0;
					printf("Warning Eye Closed,num=%d!\n", i++);
				}
			}

			visualizer.SetImage(captured_image, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
			visualizer.SetObservationLandmarks(face_model.detected_landmarks, face_model.detection_certainty, detection_success);
			visualizer.fatigueDrivingTest(face_model.detected_landmarks);

			visualizer.SetObservationPose(pose_estimate, face_model.detection_certainty);
			visualizer.SetObservationGaze(gazeDirection0, gazeDirection1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), face_model.detection_certainty);
			visualizer.SetFps(fps_tracker.GetFPS());
			// detect key presses (due to pecularities of OpenCV, you can get it when displaying images)
			char character_press = visualizer.ShowObservation();

			// restart the tracker
			if (character_press == 'r')
			{
				face_model.Reset();
			}
			// quit the application
			else if (character_press == 'q')
			{
				return(0);
			}

			// Grabbing the next frame in the sequence
			captured_image = sequence_reader.GetNextFrame();

		}
		
		// Reset the model, for the next video
		face_model.Reset();

		sequence_number++;

	}
	return 0;
}

