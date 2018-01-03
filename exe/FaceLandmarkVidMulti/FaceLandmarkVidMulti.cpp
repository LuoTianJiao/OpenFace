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
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////


// FaceTrackingVidMulti.cpp : Defines the entry point for the multiple face tracking console application.
#include "LandmarkCoreIncludes.h"

#include "VisualizationUtils.h"
#include "Visualizer.h"
#include "SequenceCapture.h"

#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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

void NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double> >& face_detections)
{

	// Go over the model and eliminate detections that are not informative (there already is a tracker there)
	for(size_t model = 0; model < clnf_models.size(); ++model)
	{

		// See if the detections intersect
		cv::Rect_<double> model_rect = clnf_models[model].GetBoundingBox();
		
		for(int detection = face_detections.size()-1; detection >=0; --detection)
		{
			double intersection_area = (model_rect & face_detections[detection]).area();
			double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;

			// If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
			if( intersection_area/union_area > 0.5)
			{
				face_detections.erase(face_detections.begin() + detection);
			}
		}
	}
}

int main (int argc, char **argv)
{

	vector<string> arguments = get_arguments(argc, argv);

			
	LandmarkDetector::FaceModelParameters det_params(arguments);
	// This is so that the model would not try re-initialising itself
	det_params.reinit_video_every = -1;

	det_params.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;

	vector<LandmarkDetector::FaceModelParameters> det_parameters;
	det_parameters.push_back(det_params);
	
	// The modules that are being used for tracking
	vector<LandmarkDetector::CLNF> face_models;
	vector<bool> active_models;

	int num_faces_max = 4;

	LandmarkDetector::CLNF face_model(det_parameters[0].model_location);
	face_model.face_detector_HAAR.load(det_parameters[0].face_detector_location);
	face_model.face_detector_location = det_parameters[0].face_detector_location;
	
	face_models.reserve(num_faces_max);

	face_models.push_back(face_model);
	active_models.push_back(false);

	for (int i = 1; i < num_faces_max; ++i)
	{
		face_models.push_back(face_model);
		active_models.push_back(false);
		det_parameters.push_back(det_params);
	}
	
	// Open a sequence
	Utilities::SequenceCapture sequence_reader;

	// A utility for visualizing the results (show just the tracks)
	Utilities::Visualizer visualizer(true, false, false);

	// Tracking FPS for visualization
	Utilities::FpsTracker fps_tracker;
	fps_tracker.AddFrame();

	int sequence_number = 0;

	while(true) // this is not a for loop as we might also be reading from a webcam
	{

		// The sequence reader chooses what to open based on command line arguments provided
		if (!sequence_reader.Open(arguments))
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

		int frame_count = 0;

		INFO_STREAM( "Starting tracking");
		while(!captured_image.empty())
		{		

			// Reading the images
			cv::Mat_<uchar> grayscale_image;

			cv::Mat disp_image = captured_image.clone();

			if(captured_image.channels() == 3)
			{
				cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);				
			}
			else
			{
				grayscale_image = captured_image.clone();				
			}
		
			vector<cv::Rect_<double> > face_detections;

			bool all_models_active = true;
			for(unsigned int model = 0; model < face_models.size(); ++model)
			{
				if(!active_models[model])
				{
					all_models_active = false;
				}
			}
						
			// Get the detections (every 8th frame and when there are free models available for tracking)
			if(frame_count % 8 == 0 && !all_models_active)
			{				
				if(det_parameters[0].curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
				{
					vector<double> confidences;
					LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, face_models[0].face_detector_HOG, confidences);
				}
				else
				{
					LandmarkDetector::DetectFaces(face_detections, grayscale_image, face_models[0].face_detector_HAAR);
				}

			}

			// Keep only non overlapping detections (also convert to a concurrent vector
			NonOverlapingDetections(face_models, face_detections);

			vector<tbb::atomic<bool> > face_detections_used(face_detections.size());

			// Go through every model and update the tracking
			tbb::parallel_for(0, (int)face_models.size(), [&](int model){
			//for(unsigned int model = 0; model < clnf_models.size(); ++model)
			//{

				bool detection_success = false;

				// If the current model has failed more than 4 times in a row, remove it
				if(face_models[model].failures_in_a_row > 4)
				{				
					active_models[model] = false;
					face_models[model].Reset();
				}

				// If the model is inactive reactivate it with new detections
				if(!active_models[model])
				{
					
					for(size_t detection_ind = 0; detection_ind < face_detections.size(); ++detection_ind)
					{
						// if it was not taken by another tracker take it (if it is false swap it to true and enter detection, this makes it parallel safe)
						if(face_detections_used[detection_ind].compare_and_swap(true, false) == false)
						{
					
							// Reinitialise the model
							face_models[model].Reset();

							// This ensures that a wider window is used for the initial landmark localisation
							face_models[model].detection_success = false;
							detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_detections[detection_ind], face_models[model], det_parameters[model]);
													
							// This activates the model
							active_models[model] = true;

							// break out of the loop as the tracker has been reinitialised
							break;
						}

					}
				}
				else
				{
					// The actual facial landmark detection / tracking
					detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_models[model], det_parameters[model]);
				}
			});
								
			// Keeping track of FPS
			fps_tracker.AddFrame();

			visualizer.SetImage(captured_image, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);

			// Go through every model and visualise the results
			for(size_t model = 0; model < face_models.size(); ++model)
			{
				// Visualising the results
				if(active_models[model])
				{
					visualizer.SetObservationLandmarks(face_models[model].detected_landmarks, face_models[model].detection_certainty, face_models[model].detection_success);
					visualizer.SetObservationPose(LandmarkDetector::GetPose(face_models[model], sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), face_models[model].detection_certainty);
				}
			}
			visualizer.SetFps(fps_tracker.GetFPS());

			// show visualization and detect key presses
			char character_press = visualizer.ShowObservation();
			
			// restart the trackers
			if(character_press == 'r')
			{
				for(size_t i=0; i < face_models.size(); ++i)
				{
					face_models[i].Reset();
					active_models[i] = false;
				}
			}
			// quit the application
			else if(character_press=='q')
			{
				return 0;
			}

			// Update the frame count
			frame_count++;

			// Grabbing the next frame in the sequence
			captured_image = sequence_reader.GetNextFrame();

		}
		
		frame_count = 0;

		// Reset the model, for the next video
		for(size_t model=0; model < face_models.size(); ++model)
		{
			face_models[model].Reset();
			active_models[model] = false;
		}

		sequence_number++;

	}

	return 0;
}

