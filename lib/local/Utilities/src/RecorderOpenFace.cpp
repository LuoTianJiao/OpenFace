///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis, all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
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

#include "RecorderOpenFace.h"

// For sorting
#include <algorithm>

// File manipulation
#include <fstream>
#include <sstream>
#include <iostream>

// Boost includes for file system manipulation
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>

using namespace boost::filesystem;

using namespace Utilities;

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

void CreateDirectory(std::string output_path)
{

	// Creating the right directory structure
	auto p = path(output_path);

	if (!boost::filesystem::exists(p))
	{
		bool success = boost::filesystem::create_directories(p);

		if (!success)
		{
			std::cout << "ERROR: failed to create output directory:" << p.string() << ", do you have permission to create directory" << std::endl;
			exit(1);
		}
	}
}

RecorderOpenFace::RecorderOpenFace(const std::string in_filename, RecorderOpenFaceParameters parameters, std::vector<std::string>& arguments):video_writer(), params(parameters)
{

	// From the filename, strip out the name without directory and extension
	if (boost::filesystem::is_directory(in_filename))
	{
		filename = boost::filesystem::canonical(boost::filesystem::path(in_filename)).filename().string();
	}
	else
	{
		filename = boost::filesystem::path(in_filename).filename().replace_extension("").string();
	}

	// Consuming the input arguments
	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-out_dir") == 0)
		{
			record_root = arguments[i + 1];
		}
	}

	// Determine output directory
	bool output_found = false;
	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (!output_found && arguments[i].compare("-of") == 0)
		{
			record_root = (boost::filesystem::path(record_root) / boost::filesystem::path(arguments[i + 1])).remove_filename().string();
			filename = path(boost::filesystem::path(arguments[i + 1])).replace_extension("").filename().string();
			valid[i] = false;
			valid[i + 1] = false;
			i++;
			output_found = true;
		}
	}

	// If recording directory not set, record to default location
	if (record_root.empty())
		record_root = default_record_directory;

	for (int i = (int)arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

	// Construct the directories required for the output
	CreateDirectory(record_root);

	// Create the filename for the general output file that contains all of the meta information about the recording
	path of_det_name(filename);
	of_det_name = path(record_root) / path(filename + "_of_details.txt");

	// Write in the of file what we are outputing what is the input etc.
	metadata_file.open(of_det_name.string(), std::ios_base::out);
	if (!metadata_file.is_open())
	{
		cout << "ERROR: could not open the output file:" << of_det_name << ", either the path of the output directory is wrong or you do not have the permissions to write to it" << endl;
		exit(1);
	}

	// Populate relative and full path names in the meta file, unless it is a webcam
	if(!params.isFromWebcam())
	{
		string input_filename_relative = in_filename;
		string input_filename_full = in_filename;
		if (!boost::filesystem::path(input_filename_full).is_absolute())
		{
			input_filename_full = boost::filesystem::canonical(input_filename_relative).string();
		}
		metadata_file << "Input:" << input_filename_relative << endl;
		metadata_file << "Input full path:" << input_filename_full << endl;
	}
	else
	{
		// Populate the metadata file
		metadata_file << "Input:webcam" << endl;
	}

	metadata_file << "Camera parameters:" << parameters.getFx() << "," << parameters.getFy() << "," << parameters.getCx() << "," << parameters.getCy() << endl;

	// Create the required individual recorders, CSV, HOG, aligned, video
	csv_filename = filename + ".csv";

	// Consruct HOG recorder here
	if(params.outputHOG())
	{
		// Output the data based on record_root, but do not include record_root in the meta file, as it is also in that directory
		std::string hog_filename = filename + ".hog";
		metadata_file << "Output HOG:" << hog_filename << endl;
		hog_filename = (path(record_root) / hog_filename).string();
		hog_recorder.Open(hog_filename);
	}

	// saving the videos	
	if (params.outputTracked())
	{
		if(parameters.isSequence())
		{
			// Output the data based on record_root, but do not include record_root in the meta file, as it is also in that directory
			this->media_filename = filename + ".avi";
			metadata_file << "Output video:" << this->media_filename << endl;
			this->media_filename = (path(record_root) / this->media_filename).string();
		}
		else
		{
			this->media_filename = filename + ".jpg";
			metadata_file << "Output image:" << this->media_filename << endl;
			this->media_filename = (path(record_root) / this->media_filename).string();
		}
	}

	// Prepare image recording
	if (params.outputAlignedFaces())
	{
		aligned_output_directory = filename + "_aligned";
		metadata_file << "Output aligned directory:" << this->aligned_output_directory << endl;
		this->aligned_output_directory = (path(record_root) / this->aligned_output_directory).string();
		CreateDirectory(aligned_output_directory);
	}
		
	observation_count = 0;

}

void RecorderOpenFace::SetObservationFaceAlign(const cv::Mat& aligned_face)
{
	this->aligned_face = aligned_face;
}

void RecorderOpenFace::SetObservationVisualization(const cv::Mat &vis_track)
{
	if (params.outputTracked())
	{
		// Initialize the video writer if it has not been opened yet
		if(params.isSequence() && !video_writer.isOpened())
		{
			std::string output_codec = params.outputCodec();
			try
			{
				video_writer.open(media_filename, CV_FOURCC(output_codec[0], output_codec[1], output_codec[2], output_codec[3]), params.outputFps(), vis_track.size(), true);

				if (!video_writer.isOpened())
				{
					WARN_STREAM("Could not open VideoWriter, OUTPUT FILE WILL NOT BE WRITTEN.");
				}
			}
			catch (cv::Exception e)
			{
				WARN_STREAM("Could not open VideoWriter, OUTPUT FILE WILL NOT BE WRITTEN. Currently using codec " << output_codec << ", try using an other one (-oc option)");
			}

		}

		vis_to_out = vis_track;

	}

}

void RecorderOpenFace::WriteObservation()
{
	observation_count++;

	// Write out the CSV file (it will always be there, even if not outputting anything more but frame/face numbers)
	
	if(observation_count == 1)
	{
		// As we are writing out the header, work out some things like number of landmarks, names of AUs etc.
		int num_face_landmarks = landmarks_2D.rows / 2;
		int num_eye_landmarks = (int)eye_landmarks2D.size();
		int num_model_modes = pdm_params_local.rows;

		std::vector<std::string> au_names_class;
		for (auto au : au_occurences)
		{
			au_names_class.push_back(au.first);
		}

		std::sort(au_names_class.begin(), au_names_class.end());

		std::vector<std::string> au_names_reg;
		for (auto au : au_intensities)
		{
			au_names_reg.push_back(au.first);
		}

		std::sort(au_names_reg.begin(), au_names_reg.end());

		metadata_file << "Output csv:" << csv_filename << endl;
		metadata_file << "Gaze: " << params.outputGaze() << endl;
		metadata_file << "AUs: " << params.outputAUs() << endl;
		metadata_file << "Landmarks 2D: " << params.output2DLandmarks() << endl;
		metadata_file << "Landmarks 3D: " << params.output3DLandmarks() << endl;
		metadata_file << "Pose: " << params.outputPose() << endl;
		metadata_file << "Shape parameters: " << params.outputPDMParams() << endl;

		csv_filename = (path(record_root) / csv_filename).string();
		csv_recorder.Open(csv_filename, params.isSequence(), params.output2DLandmarks(), params.output3DLandmarks(), params.outputPDMParams(), params.outputPose(),
			params.outputAUs(), params.outputGaze(), num_face_landmarks, num_model_modes, num_eye_landmarks, au_names_class, au_names_reg);
	}

	this->csv_recorder.WriteLine(observation_count, timestamp, landmark_detection_success, 
		landmark_detection_confidence, landmarks_2D, landmarks_3D, pdm_params_local, pdm_params_global, head_pose,
		gaze_direction0, gaze_direction1, gaze_angle, eye_landmarks2D, eye_landmarks3D, au_intensities, au_occurences);

	if(params.outputHOG())
	{
		this->hog_recorder.Write();
	}

	// Write aligned faces
	if (params.outputAlignedFaces())
	{
		char name[100];

		// Filename is based on frame number
		if(params.isSequence())
			std::sprintf(name, "frame_det_%06d.bmp", observation_count);
		else
			std::sprintf(name, "face_det_%06d.bmp", observation_count);

		// Construct the output filename
		boost::filesystem::path slash("/");

		std::string preferredSlash = slash.make_preferred().string();

		string out_file = aligned_output_directory + preferredSlash + string(name);
		bool write_success = cv::imwrite(out_file, aligned_face);

		if (!write_success)
		{
			WARN_STREAM("Could not output similarity aligned image image");
		}
	}

	if(params.outputTracked())
	{
		if (vis_to_out.empty())
		{
			WARN_STREAM("Output tracked video frame is not set");
		}

		if(params.isSequence() )
		{
			if(video_writer.isOpened())
			{
				video_writer.write(vis_to_out);
			}
		}
		else
		{
			bool out_success = cv::imwrite(media_filename, vis_to_out);
			if (!out_success)
			{
				WARN_STREAM("Could not output tracked image");
			}
		}
		// Clear the output
		vis_to_out = cv::Mat();
	}
}


void RecorderOpenFace::SetObservationHOG(bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_cols, int num_rows, int num_channels)
{
	this->hog_recorder.SetObservationHOG(good_frame, hog_descriptor, num_cols, num_rows, num_channels);
}

void RecorderOpenFace::SetObservationTimestamp(double timestamp)
{
	this->timestamp = timestamp;
}

void RecorderOpenFace::SetObservationLandmarks(const cv::Mat_<double>& landmarks_2D, const cv::Mat_<double>& landmarks_3D,
	const cv::Vec6d& pdm_params_global, const cv::Mat_<double>& pdm_params_local, double confidence, bool success)
{
	this->landmarks_2D = landmarks_2D;
	this->landmarks_3D = landmarks_3D;
	this->pdm_params_global = pdm_params_global;
	this->pdm_params_local = pdm_params_local;
	this->landmark_detection_confidence = confidence;
	this->landmark_detection_success = success;

}

void RecorderOpenFace::SetObservationPose(const cv::Vec6d& pose)
{
	this->head_pose = pose;
}

void RecorderOpenFace::SetObservationActionUnits(const std::vector<std::pair<std::string, double> >& au_intensities,
	const std::vector<std::pair<std::string, double> >& au_occurences)
{
	this->au_intensities = au_intensities;
	this->au_occurences = au_occurences;
}

void RecorderOpenFace::SetObservationGaze(const cv::Point3f& gaze_direction0, const cv::Point3f& gaze_direction1,
	const cv::Vec2d& gaze_angle, const std::vector<cv::Point2d>& eye_landmarks2D, const std::vector<cv::Point3d>& eye_landmarks3D)
{
	this->gaze_direction0 = gaze_direction0;
	this->gaze_direction1 = gaze_direction1;
	this->gaze_angle = gaze_angle;
	this->eye_landmarks2D = eye_landmarks2D;
	this->eye_landmarks3D = eye_landmarks3D;
}

RecorderOpenFace::~RecorderOpenFace()
{
	this->Close();
}


void RecorderOpenFace::Close()
{
	hog_recorder.Close();
	csv_recorder.Close();
	video_writer.release();
	metadata_file.close();

}



