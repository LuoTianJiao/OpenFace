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

#include "Visualizer.h"
#include "VisualizationUtils.h"
#include "RotationHelpers.h"
#include "ImageManipulationHelpers.h"

// For drawing on images
#include <opencv2/imgproc.hpp>

using namespace Utilities;

// For subpixel accuracy drawing
const int draw_shiftbits = 4;
const int draw_multiplier = 1 << 4;

Visualizer::Visualizer(std::vector<std::string> arguments)
{
	// By default not visualizing anything
	this->vis_track = false;
	this->vis_hog = false;
	this->vis_align = false;

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-verbose") == 0)
		{
			vis_track = true;
			vis_align = true;
			vis_hog = true;
		}
		else if (arguments[i].compare("-vis-align") == 0)
		{
			vis_align = true;
		}
		else if (arguments[i].compare("-vis-hog") == 0)
		{
			vis_hog = true;
		}
		else if (arguments[i].compare("-vis-track") == 0)
		{
			vis_track = true;
		}
	}

}

Visualizer::Visualizer(bool vis_track, bool vis_hog, bool vis_align)
{
	this->vis_track = vis_track;
	this->vis_hog = vis_hog;
	this->vis_align = vis_align;
}



// Setting the image on which to draw
void Visualizer::SetImage(const cv::Mat& canvas, float fx, float fy, float cx, float cy)
{
	// Convert the image to 8 bit RGB
	captured_image = canvas.clone();

	this->fx = fx;
	this->fy = fy;
	this->cx = cx;
	this->cy = cy;

	// Clearing other images
	hog_image = cv::Mat();
	aligned_face_image = cv::Mat();

}


void Visualizer::SetObservationFaceAlign(const cv::Mat& aligned_face)
{
	if(this->aligned_face_image.empty())
	{
		this->aligned_face_image = aligned_face;
	}
	else
	{
		cv::vconcat(this->aligned_face_image, aligned_face, this->aligned_face_image);
	}
}

void Visualizer::SetObservationHOG(const cv::Mat_<double>& hog_descriptor, int num_cols, int num_rows)
{
	if(vis_hog)
	{
		if (this->hog_image.empty())
		{
			Visualise_FHOG(hog_descriptor, num_rows, num_cols, this->hog_image);
		}
		else
		{
			cv::Mat tmp_hog;
			Visualise_FHOG(hog_descriptor, num_rows, num_cols, tmp_hog);
			cv::vconcat(this->hog_image, tmp_hog, this->hog_image);
		}
	}

}


void Visualizer::SetObservationLandmarks(const cv::Mat_<double>& landmarks_2D, double confidence, bool success, const cv::Mat_<int>& visibilities)
{

	if(confidence > visualisation_boundary)
	{
		// Draw 2D landmarks on the image
		int n = landmarks_2D.rows / 2;

		// Drawing feature points
		for (int i = 0; i < n; ++i)
		{
			if (visibilities.empty() || visibilities.at<int>(i))
			{
				cv::Point featurePoint(cvRound(landmarks_2D.at<double>(i) * (double)draw_multiplier), cvRound(landmarks_2D.at<double>(i + n) * (double)draw_multiplier));

				// A rough heuristic for drawn point size
				int thickness = (int)std::ceil(3.0* ((double)captured_image.cols) / 640.0);
				int thickness_2 = (int)std::ceil(1.0* ((double)captured_image.cols) / 640.0);

				cv::circle(captured_image, featurePoint, 1 * draw_multiplier, cv::Scalar(0, 0, 255), thickness, CV_AA, draw_shiftbits);
				cv::circle(captured_image, featurePoint, 1 * draw_multiplier, cv::Scalar(255, 0, 0), thickness_2, CV_AA, draw_shiftbits);

			}
		}
	}
}

void Visualizer::SetObservationPose(const cv::Vec6d& pose, double confidence)
{



	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (confidence > visualisation_boundary)
	{
		double vis_certainty = confidence;
		if (vis_certainty > 1)
			vis_certainty = 1;

		// Scale from 0 to 1, to allow to indicated by colour how confident we are in the tracking
		vis_certainty = (vis_certainty - visualisation_boundary) / (1 - visualisation_boundary);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		// Draw it in reddish if uncertain, blueish if certain
		DrawBox(captured_image, pose, cv::Scalar(vis_certainty*255.0, 0, (1 - vis_certainty) * 255), thickness, fx, fy, cx, cy);
	}
}

// Eye gaze infomration drawing, first of eye landmarks then of gaze
void Visualizer::SetObservationGaze(const cv::Point3f& gaze_direction0, const cv::Point3f& gaze_direction1, const std::vector<cv::Point2d>& eye_landmarks2d, const std::vector<cv::Point3d>& eye_landmarks3d, double confidence)
{
	if(confidence > visualisation_boundary)
	{
		if (eye_landmarks2d.size() > 0)
		{
			// First draw the eye region landmarks
			for (size_t i = 0; i < eye_landmarks2d.size(); ++i)
			{
				cv::Point featurePoint(cvRound(eye_landmarks2d[i].x * (double)draw_multiplier), cvRound(eye_landmarks2d[i].y * (double)draw_multiplier));

				// A rough heuristic for drawn point size
				int thickness = 1;
				int thickness_2 = 1;

				size_t next_point = i + 1;
				if (i == 7)
					next_point = 0;
				if (i == 19)
					next_point = 8;
				if (i == 27)
					next_point = 20;

				if (i == 7 + 28)
					next_point = 0 + 28;
				if (i == 19 + 28)
					next_point = 8 + 28;
				if (i == 27 + 28)
					next_point = 20 + 28;

				cv::Point nextFeaturePoint(cvRound(eye_landmarks2d[next_point].x * (double)draw_multiplier), cvRound(eye_landmarks2d[next_point].y * (double)draw_multiplier));
				if ((i < 28 && (i < 8 || i > 19)) || (i >= 28 && (i < 8 + 28 || i > 19 + 28)))
					cv::line(captured_image, featurePoint, nextFeaturePoint, cv::Scalar(255, 0, 0), thickness_2, CV_AA, draw_shiftbits);
				else
					cv::line(captured_image, featurePoint, nextFeaturePoint, cv::Scalar(0, 0, 255), thickness_2, CV_AA, draw_shiftbits);

			}

			// Now draw the gaze lines themselves
			cv::Mat cameraMat = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 0);

			// Grabbing the pupil location, to draw eye gaze need to know where the pupil is
			cv::Point3d pupil_left(0, 0, 0);
			cv::Point3d pupil_right(0, 0, 0);
			for (size_t i = 0; i < 8; ++i)
			{
				pupil_left = pupil_left + eye_landmarks3d[i];
				pupil_right = pupil_right + eye_landmarks3d[i + eye_landmarks3d.size()/2];
			}
			pupil_left = pupil_left / 8;
			pupil_right = pupil_right / 8;

			std::vector<cv::Point3d> points_left;
			points_left.push_back(cv::Point3d(pupil_left));
			points_left.push_back(cv::Point3d(pupil_left + cv::Point3d(gaze_direction0)*50.0));

			std::vector<cv::Point3d> points_right;
			points_right.push_back(cv::Point3d(pupil_right));
			points_right.push_back(cv::Point3d(pupil_right + cv::Point3d(gaze_direction1)*50.0));

			cv::Mat_<double> proj_points;
			cv::Mat_<double> mesh_0 = (cv::Mat_<double>(2, 3) << points_left[0].x, points_left[0].y, points_left[0].z, points_left[1].x, points_left[1].y, points_left[1].z);
			Project(proj_points, mesh_0, fx, fy, cx, cy);
			cv::line(captured_image, cv::Point(cvRound(proj_points.at<double>(0, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(0, 1) * (double)draw_multiplier)),
				cv::Point(cvRound(proj_points.at<double>(1, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(1, 1) * (double)draw_multiplier)), cv::Scalar(110, 220, 0), 2, CV_AA, draw_shiftbits);

			cv::Mat_<double> mesh_1 = (cv::Mat_<double>(2, 3) << points_right[0].x, points_right[0].y, points_right[0].z, points_right[1].x, points_right[1].y, points_right[1].z);
			Project(proj_points, mesh_1, fx, fy, cx, cy);
			cv::line(captured_image, cv::Point(cvRound(proj_points.at<double>(0, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(0, 1) * (double)draw_multiplier)),
				cv::Point(cvRound(proj_points.at<double>(1, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(1, 1) * (double)draw_multiplier)), cv::Scalar(110, 220, 0), 2, CV_AA, draw_shiftbits);

		}
	}
}

void Visualizer::SetFps(double fps)
{
	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps);
	std::string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);
}

char Visualizer::ShowObservation()
{
	if (vis_track)
	{
		cv::namedWindow("tracking_result", 1);
		cv::imshow("tracking_result", captured_image);
	}
	if (vis_align)
	{
		cv::imshow("sim_warp", aligned_face_image);
	}
	if (vis_hog)
	{
		cv::imshow("hog", hog_image);
	}
	return cv::waitKey(1);
}

cv::Mat Visualizer::GetVisImage()
{
	return captured_image;
}
