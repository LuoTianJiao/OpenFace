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

#include "RecorderOpenFaceParameters.h"

using namespace std;

using namespace Utilities;

RecorderOpenFaceParameters::RecorderOpenFaceParameters(std::vector<std::string> &arguments, bool sequence, bool from_webcam, float fx, float fy, float cx, float cy, double fps_vid_out)
{

	string separator = string(1, boost::filesystem::path::preferred_separator);

	this->is_sequence = sequence;
	this->is_from_webcam = from_webcam;
	this->fx = fx;
	this->fy = fy;
	this->cx = cx;
	this->cy = cy;

	if(fps_vid_out > 0)
	{
		this->fps_vid_out = fps_vid_out;
	}
	else
	{
		this->fps_vid_out = 30; // If an illegal value for fps provided, default to 30
	}
	// Default output code
	this->output_codec = "DIVX";

	bool output_set = false;

	output_2D_landmarks = false;
	output_3D_landmarks = false;
	output_model_params = false;
	output_pose = false;
	output_AUs = false;
	output_gaze = false;
	output_hog = false;
	output_tracked = false;
	output_aligned_faces = false;

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-simalign") == 0)
		{
			output_aligned_faces = true;
			output_set = true;
		}
		else if (arguments[i].compare("-hogalign") == 0)
		{
			output_hog = true;
			output_set = true;
		}
		else if (arguments[i].compare("-2Dfp") == 0)
		{
			output_2D_landmarks = true;
			output_set = true;
		}
		else if (arguments[i].compare("-3Dfp") == 0)
		{
			output_3D_landmarks = true;
			output_set = true;
		}
		else if (arguments[i].compare("-pdmparams") == 0)
		{
			output_model_params = true;
			output_set = true;
		}
		else if (arguments[i].compare("-pose") == 0)
		{
			output_pose = true;
			output_set = true;
		}
		else if (arguments[i].compare("-aus") == 0)
		{
			output_AUs = true;
			output_set = true;
		}
		else if (arguments[i].compare("-gaze") == 0)
		{
			output_gaze = true;
			output_set = true;
		}
		else if (arguments[i].compare("-tracked") == 0)
		{
			output_tracked = true;
			output_set = true;
		}
	}

	// Output everything if nothing has been set

	if (!output_set)
	{
		output_2D_landmarks = true;
		output_3D_landmarks = true;
		output_model_params = true;
		output_pose = true;
		output_AUs = true;
		output_gaze = true;
		output_hog = true;
		output_tracked = true;
		output_aligned_faces = true;
	}

}

