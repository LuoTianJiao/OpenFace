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

#include "stdafx.h"

#include <LandmarkDetectorUtils.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

namespace LandmarkDetector
{

//===========================================================================
// Fast patch expert response computation (linear model across a ROI) using normalised cross-correlation
//===========================================================================

void crossCorr_m( const cv::Mat_<float>& img, cv::Mat_<double>& img_dft, const cv::Mat_<float>& _templ, map<int, cv::Mat_<double> >& _templ_dfts, cv::Mat_<float>& corr)
{
	// Our model will always be under min block size so can ignore this
    //const double blockScale = 4.5;
    //const int minBlockSize = 256;

	int maxDepth = CV_64F;

	cv::Size dftsize;
	
    dftsize.width = cv::getOptimalDFTSize(corr.cols + _templ.cols - 1);
    dftsize.height = cv::getOptimalDFTSize(corr.rows + _templ.rows - 1);

    // Compute block size
	cv::Size blocksize;
    blocksize.width = dftsize.width - _templ.cols + 1;
    blocksize.width = MIN( blocksize.width, corr.cols );
    blocksize.height = dftsize.height - _templ.rows + 1;
    blocksize.height = MIN( blocksize.height, corr.rows );
	
	cv::Mat_<double> dftTempl;

	// if this has not been precomputed, precompute it, otherwise use it
	if(_templ_dfts.find(dftsize.width) == _templ_dfts.end())
	{
		dftTempl.create(dftsize.height, dftsize.width);

		cv::Mat_<float> src = _templ;

		cv::Mat_<double> dst(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));
		
		cv::Mat_<double> dst1(dftTempl, cv::Rect(0, 0, _templ.cols, _templ.rows));
			
		if( dst1.data != src.data )
			src.convertTo(dst1, dst1.depth());

		if( dst.cols > _templ.cols )
		{
			cv::Mat_<double> part(dst, cv::Range(0, _templ.rows), cv::Range(_templ.cols, dst.cols));
			part.setTo(0);
		}

		// Perform DFT of the template
		dft(dst, dst, 0, _templ.rows);
		
		_templ_dfts[dftsize.width] = dftTempl;

	}
	else
	{
		// use the precomputed version
		dftTempl = _templ_dfts.find(dftsize.width)->second;
	}

	cv::Size bsz(std::min(blocksize.width, corr.cols), std::min(blocksize.height, corr.rows));
	cv::Mat src;

	cv::Mat cdst(corr, cv::Rect(0, 0, bsz.width, bsz.height));
	
	cv::Mat_<double> dftImg;

	if(img_dft.empty())
	{
		dftImg.create(dftsize);
		dftImg.setTo(0.0);

		cv::Size dsz(bsz.width + _templ.cols - 1, bsz.height + _templ.rows - 1);

		int x2 = std::min(img.cols, dsz.width);
		int y2 = std::min(img.rows, dsz.height);

		cv::Mat src0(img, cv::Range(0, y2), cv::Range(0, x2));
		cv::Mat dst(dftImg, cv::Rect(0, 0, dsz.width, dsz.height));
		cv::Mat dst1(dftImg, cv::Rect(0, 0, x2, y2));

		src = src0;
		
		if( dst1.data != src.data )
			src.convertTo(dst1, dst1.depth());

		dft( dftImg, dftImg, 0, dsz.height );
		img_dft = dftImg.clone();
	}

	cv::Mat dftTempl1(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));
	cv::mulSpectrums(img_dft, dftTempl1, dftImg, 0, true);
	cv::dft( dftImg, dftImg, cv::DFT_INVERSE + cv::DFT_SCALE, bsz.height );

	src = dftImg(cv::Rect(0, 0, bsz.width, bsz.height));

	src.convertTo(cdst, CV_32F);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////

void matchTemplate_m(  const cv::Mat_<float>& input_img, cv::Mat_<double>& img_dft, cv::Mat& _integral_img, cv::Mat& _integral_img_sq, const cv::Mat_<float>&  templ, map<int, cv::Mat_<double> >& templ_dfts, cv::Mat_<float>& result, int method )
{

        int numType = method == CV_TM_CCORR || method == CV_TM_CCORR_NORMED ? 0 :
                  method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED ? 1 : 2;
    bool isNormed = method == CV_TM_CCORR_NORMED ||
                    method == CV_TM_SQDIFF_NORMED ||
                    method == CV_TM_CCOEFF_NORMED;
	
	// Assume result is defined properly
	if(result.empty())
	{
		cv::Size corrSize(input_img.cols - templ.cols + 1, input_img.rows - templ.rows + 1);
		result.create(corrSize);
	}
    LandmarkDetector::crossCorr_m( input_img, img_dft, templ, templ_dfts, result);

    if( method == CV_TM_CCORR )
        return;

    double invArea = 1./((double)templ.rows * templ.cols);

	cv::Mat sum, sqsum;
	cv::Scalar templMean, templSdv;
    double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
    double templNorm = 0, templSum2 = 0;

    if( method == CV_TM_CCOEFF )
    {
		// If it has not been precomputed compute it now
		if(_integral_img.empty())
		{
			integral(input_img, _integral_img, CV_64F);
		}
		sum = _integral_img;

        templMean = cv::mean(templ);
    }
    else
    {
		// If it has not been precomputed compute it now
		if(_integral_img.empty())
		{
			integral(input_img, _integral_img, _integral_img_sq, CV_64F);			
		}

		sum = _integral_img;
		sqsum = _integral_img_sq;

        meanStdDev( templ, templMean, templSdv );

        templNorm = templSdv[0]*templSdv[0] + templSdv[1]*templSdv[1] + templSdv[2]*templSdv[2] + templSdv[3]*templSdv[3];

        if( templNorm < DBL_EPSILON && method == CV_TM_CCOEFF_NORMED )
        {
			result.setTo(1.0);
            return;
        }

        templSum2 = templNorm + templMean[0]*templMean[0] + templMean[1]*templMean[1] + templMean[2]*templMean[2] + templMean[3]*templMean[3];

        if( numType != 1 )
        {
            templMean = cv::Scalar::all(0);
            templNorm = templSum2;
        }

        templSum2 /= invArea;
        templNorm = std::sqrt(templNorm);
        templNorm /= std::sqrt(invArea); // care of accuracy here

        q0 = (double*)sqsum.data;
        q1 = q0 + templ.cols;
        q2 = (double*)(sqsum.data + templ.rows*sqsum.step);
        q3 = q2 + templ.cols;
    }

    double* p0 = (double*)sum.data;
    double* p1 = p0 + templ.cols;
    double* p2 = (double*)(sum.data + templ.rows*sum.step);
    double* p3 = p2 + templ.cols;

    int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
    int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

    int i, j;

    for( i = 0; i < result.rows; i++ )
    {
        float* rrow = result.ptr<float>(i);
        int idx = i * sumstep;
        int idx2 = i * sqstep;

        for( j = 0; j < result.cols; j++, idx += 1, idx2 += 1 )
        {
            double num = rrow[j], t;
            double wndMean2 = 0, wndSum2 = 0;

            if( numType == 1 )
            {

                t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
                wndMean2 += t*t;
                num -= t*templMean[0];

                wndMean2 *= invArea;
            }

            if( isNormed || numType == 2 )
            {

                t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
                wndSum2 += t;

                if( numType == 2 )
                {
                    num = wndSum2 - 2*num + templSum2;
                    num = MAX(num, 0.);
                }
            }

            if( isNormed )
            {
                t = std::sqrt(MAX(wndSum2 - wndMean2,0))*templNorm;
                if( fabs(num) < t )
                    num /= t;
                else if( fabs(num) < t*1.125 )
                    num = num > 0 ? 1 : -1;
                else
                    num = method != CV_TM_SQDIFF_NORMED ? 0 : 1;
            }

            rrow[j] = (float)num;
        }
    }
}


//===========================================================================
// Point set and landmark manipulation functions
//===========================================================================
// Using Kabsch's algorithm for aligning shapes
//This assumes that align_from and align_to are already mean normalised
cv::Matx22d AlignShapesKabsch2D(const cv::Mat_<double>& align_from, const cv::Mat_<double>& align_to )
{

	cv::SVD svd(align_from.t() * align_to);
    
	// make sure no reflection is there
	// corr ensures that we do only rotaitons and not reflections
	double d = cv::determinant(svd.vt.t() * svd.u.t());

	cv::Matx22d corr = cv::Matx22d::eye();
	if(d > 0)
	{
		corr(1,1) = 1;
	}
	else
	{
		corr(1,1) = -1;
	}

	cv::Matx22d R;
	cv::Mat(svd.vt.t()*cv::Mat(corr)*svd.u.t()).copyTo(R);

	return R;
}

//=============================================================================
// Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
cv::Matx22d AlignShapesWithScale(cv::Mat_<double>& src, cv::Mat_<double> dst)
{
	int n = src.rows;

	// First we mean normalise both src and dst
	double mean_src_x = cv::mean(src.col(0))[0];
	double mean_src_y = cv::mean(src.col(1))[0];

	double mean_dst_x = cv::mean(dst.col(0))[0];
	double mean_dst_y = cv::mean(dst.col(1))[0];

	cv::Mat_<double> src_mean_normed = src.clone();
	src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
	src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;

	cv::Mat_<double> dst_mean_normed = dst.clone();
	dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
	dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;

	// Find the scaling factor of each
	cv::Mat src_sq;
	cv::pow(src_mean_normed, 2, src_sq);

	cv::Mat dst_sq;
	cv::pow(dst_mean_normed, 2, dst_sq);

	double s_src = sqrt(cv::sum(src_sq)[0]/n);
	double s_dst = sqrt(cv::sum(dst_sq)[0]/n);

	src_mean_normed = src_mean_normed / s_src;
	dst_mean_normed = dst_mean_normed / s_dst;

	double s = s_dst / s_src;

	// Get the rotation
	cv::Matx22d R = AlignShapesKabsch2D(src_mean_normed, dst_mean_normed);
		
	cv::Matx22d	A;
	cv::Mat(s * R).copyTo(A);

	cv::Mat_<double> aligned = (cv::Mat(cv::Mat(A) * src.t())).t();
	cv::Mat_<double> offset = dst - aligned;

	double t_x =  cv::mean(offset.col(0))[0];
	double t_y =  cv::mean(offset.col(1))[0];
    
	return A;

}


//===========================================================================
// Visualisation functions
//===========================================================================

// Computing landmarks (to be drawn later possibly)
vector<cv::Point2d> CalculateVisibleLandmarks(const cv::Mat_<double>& shape2D, const cv::Mat_<int>& visibilities)
{
	int n = shape2D.rows / 2;
	vector<cv::Point2d> landmarks;

	for (int i = 0; i < n; ++i)
	{
		if (visibilities.at<int>(i))
		{
			cv::Point2d featurePoint(shape2D.at<double>(i), shape2D.at<double>(i + n));

			landmarks.push_back(featurePoint);
		}
	}

	return landmarks;
}

// Computing landmarks (to be drawn later possibly)
vector<cv::Point2d> CalculateAllLandmarks(const cv::Mat_<double>& shape2D)
{

	int n;
	vector<cv::Point2d> landmarks;

	if (shape2D.cols == 2)
	{
		n = shape2D.rows;
	}
	else if (shape2D.cols == 1)
	{
		n = shape2D.rows / 2;
	}

	for (int i = 0; i < n; ++i)
	{
		cv::Point2d featurePoint;
		if (shape2D.cols == 1)
		{
			featurePoint = cv::Point2d(shape2D.at<double>(i), shape2D.at<double>(i + n));
		}
		else
		{
			featurePoint = cv::Point2d(shape2D.at<double>(i, 0), shape2D.at<double>(i, 1));
		}

		landmarks.push_back(featurePoint);
	}

	return landmarks;
}

// Computing landmarks (to be drawn later possibly)
vector<cv::Point2d> CalculateAllLandmarks(const CLNF& clnf_model)
{
	return CalculateAllLandmarks(clnf_model.detected_landmarks);
}

// Computing landmarks (to be drawn later possibly)
vector<cv::Point2d> CalculateVisibleLandmarks(const CLNF& clnf_model)
{
	// If the detection was not successful no landmarks are visible
	if (clnf_model.detection_success)
	{
		int idx = clnf_model.patch_experts.GetViewIdx(clnf_model.params_global, 0);
		// Because we may want to draw visible points, need to find which points patch experts consider visible at a certain orientation
		return CalculateVisibleLandmarks(clnf_model.detected_landmarks, clnf_model.patch_experts.visibilities[0][idx]);
	}
	else
	{
		return vector<cv::Point2d>();
	}
}

// Computing eye landmarks
vector<cv::Point2d> CalculateVisibleEyeLandmarks(const CLNF& clnf_model)
{

	vector<cv::Point2d> to_return;

	for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
	{

		if (clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0 ||
			clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
		{

			auto lmks = CalculateVisibleLandmarks(clnf_model.hierarchical_models[i]);
			for (auto lmk : lmks)
			{
				to_return.push_back(lmk);
			}
		}
	}
	return to_return;
}

// Computing the 3D eye landmarks
vector<cv::Point3d> Calculate3DEyeLandmarks(const CLNF& clnf_model, double fx, double fy, double cx, double cy)
{

	vector<cv::Point3d> to_return;

	for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
	{

		if (clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0 ||
			clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
		{
			
			auto lmks = clnf_model.hierarchical_models[i].GetShape(fx, fy, cx, cy);

			int num_landmarks = lmks.cols;

			for (int lmk = 0; lmk < num_landmarks; ++lmk)
			{
				cv::Point3d curr_lmk(lmks.at<double>(0, lmk), lmks.at<double>(1, lmk), lmks.at<double>(2, lmk));
				to_return.push_back(curr_lmk);
			}
		}
	}
	return to_return;
}
// Computing eye landmarks
vector<cv::Point2d> CalculateAllEyeLandmarks(const CLNF& clnf_model)
{

	vector<cv::Point2d> to_return;

	for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
	{

		if (clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0 ||
			clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
		{

			auto lmks = CalculateAllLandmarks(clnf_model.hierarchical_models[i]);
			for (auto lmk : lmks)
			{
				to_return.push_back(lmk);
			}
		}
	}
	return to_return;
}

//===========================================================================

//============================================================================
// Face detection helpers
//============================================================================
bool DetectFaces(vector<cv::Rect_<double> >& o_regions, const cv::Mat_<uchar>& intensity)
{
	cv::CascadeClassifier classifier("./classifiers/haarcascade_frontalface_alt.xml");
	if(classifier.empty())
	{
		cout << "Couldn't load the Haar cascade classifier" << endl;
		return false;
	}
	else
	{
		return DetectFaces(o_regions, intensity, classifier);
	}

}

bool DetectFaces(vector<cv::Rect_<double> >& o_regions, const cv::Mat_<uchar>& intensity, cv::CascadeClassifier& classifier)
{
		
	vector<cv::Rect> face_detections;
	classifier.detectMultiScale(intensity, face_detections, 1.2, 2, 0, cv::Size(50, 50));

	// Convert from int bounding box do a double one with corrections
	o_regions.resize(face_detections.size());

	for( size_t face = 0; face < o_regions.size(); ++face)
	{
		// OpenCV is overgenerous with face size and y location is off
		// CLNF detector expects the bounding box to encompass from eyebrow to chin in y, and from cheeck outline to cheeck outline in x, so we need to compensate

		// The scalings were learned using the Face Detections on LFPW, Helen, AFW and iBUG datasets, using ground truth and detections from openCV

		// Correct for scale
		o_regions[face].width = face_detections[face].width * 0.8924; 
		o_regions[face].height = face_detections[face].height * 0.8676;

		// Move the face slightly to the right (as the width was made smaller)
		o_regions[face].x = face_detections[face].x + 0.0578 * face_detections[face].width;
		// Shift face down as OpenCV Haar Cascade detects the forehead as well, and we're not interested
		o_regions[face].y = face_detections[face].y + face_detections[face].height * 0.2166;
		
		
	}
	return o_regions.size() > 0;
}

bool DetectSingleFace(cv::Rect_<double>& o_region, const cv::Mat_<uchar>& intensity_image, cv::CascadeClassifier& classifier, cv::Point preference)
{
	// The tracker can return multiple faces
	vector<cv::Rect_<double> > face_detections;
				
	bool detect_success = LandmarkDetector::DetectFaces(face_detections, intensity_image, classifier);
					
	if(detect_success)
	{
		
		bool use_preferred = (preference.x != -1) && (preference.y != -1);

		if(face_detections.size() > 1)
		{
			// keep the closest one if preference point not set
			double best = -1;
			int bestIndex = -1;
			for( size_t i = 0; i < face_detections.size(); ++i)
			{
				double dist;
				bool better;

				if(use_preferred)
				{
					dist = sqrt((preference.x) * (face_detections[i].width/2 + face_detections[i].x) + 
								(preference.y) * (face_detections[i].height/2 + face_detections[i].y));
					better = dist < best;
				}
				else
				{
					dist = face_detections[i].width;
					better = face_detections[i].width > best;
				}

				// Pick a closest face to preffered point or the biggest face
				if(i == 0 || better)
				{
					bestIndex = i;	
					best = dist;
				}									
			}

			o_region = face_detections[bestIndex];

		}
		else
		{	
			o_region = face_detections[0];
		}				
	
	}
	else
	{
		// if not detected
		o_region = cv::Rect_<double>(0,0,0,0);
	}
	return detect_success;
}

bool DetectFacesHOG(vector<cv::Rect_<double> >& o_regions, const cv::Mat_<uchar>& intensity, std::vector<double>& confidences)
{
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

	return DetectFacesHOG(o_regions, intensity, detector, confidences);

}

bool DetectFacesHOG(vector<cv::Rect_<double> >& o_regions, const cv::Mat_<uchar>& intensity, dlib::frontal_face_detector& detector, std::vector<double>& o_confidences)
{
		
	cv::Mat_<uchar> upsampled_intensity;

	double scaling = 1.3;

	cv::resize(intensity, upsampled_intensity, cv::Size((int)(intensity.cols * scaling), (int)(intensity.rows * scaling)));

	dlib::cv_image<uchar> cv_grayscale(upsampled_intensity);

	std::vector<dlib::full_detection> face_detections;
	detector(cv_grayscale, face_detections, -0.2);

	// Convert from int bounding box do a double one with corrections
	o_regions.resize(face_detections.size());
	o_confidences.resize(face_detections.size());

	for( size_t face = 0; face < o_regions.size(); ++face)
	{
		// CLNF expects the bounding box to encompass from eyebrow to chin in y, and from cheeck outline to cheeck outline in x, so we need to compensate

		// The scalings were learned using the Face Detections on LFPW and Helen using ground truth and detections from the HOG detector

		// Move the face slightly to the right (as the width was made smaller)
		o_regions[face].x = (face_detections[face].rect.get_rect().tl_corner().x() + 0.0389 * face_detections[face].rect.get_rect().width())/scaling;
		// Shift face down as OpenCV Haar Cascade detects the forehead as well, and we're not interested
		o_regions[face].y = (face_detections[face].rect.get_rect().tl_corner().y() + 0.1278 * face_detections[face].rect.get_rect().height())/scaling;

		// Correct for scale
		o_regions[face].width = (face_detections[face].rect.get_rect().width() * 0.9611)/scaling; 
		o_regions[face].height = (face_detections[face].rect.get_rect().height() * 0.9388)/scaling;

		o_confidences[face] = face_detections[face].detection_confidence;
		
		
	}
	return o_regions.size() > 0;
}

bool DetectSingleFaceHOG(cv::Rect_<double>& o_region, const cv::Mat_<uchar>& intensity_img, dlib::frontal_face_detector& detector, double& confidence, cv::Point preference)
{
	// The tracker can return multiple faces
	vector<cv::Rect_<double> > face_detections;
	vector<double> confidences;

	bool detect_success = LandmarkDetector::DetectFacesHOG(face_detections, intensity_img, detector, confidences);
					
	if(detect_success)
	{

		bool use_preferred = (preference.x != -1) && (preference.y != -1);

		// keep the most confident one or the one closest to preference point if set
		double best_so_far;
		if(use_preferred)
		{			
			best_so_far = sqrt((preference.x - (face_detections[0].width/2 + face_detections[0].x)) * (preference.x - (face_detections[0].width/2 + face_detections[0].x)) + 
							   (preference.y - (face_detections[0].height/2 + face_detections[0].y)) * (preference.y - (face_detections[0].height/2 + face_detections[0].y)));
		}
		else
		{
			best_so_far = confidences[0];
		}
		int bestIndex = 0;

		for( size_t i = 1; i < face_detections.size(); ++i)
		{

			double dist;
			bool better;

			if(use_preferred)
			{
				dist = sqrt((preference.x - (face_detections[i].width/2 + face_detections[i].x)) * (preference.x - (face_detections[i].width/2 + face_detections[i].x)) + 
							   (preference.y - (face_detections[i].height/2 + face_detections[i].y)) * (preference.y - (face_detections[i].height/2 + face_detections[i].y)));
				better = dist < best_so_far;
			}
			else
			{
				dist = confidences[i];
				better = dist > best_so_far;
			}

			// Pick a closest face
			if(better)
			{
				best_so_far = dist;
				bestIndex = i;
			}									
		}

		o_region = face_detections[bestIndex];
		confidence = confidences[bestIndex];
	}
	else
	{
		// if not detected
		o_region = cv::Rect_<double>(0,0,0,0);
		// A completely unreliable detection (shouldn't really matter what is returned here)
		confidence = -2;		
	}
	return detect_success;
}

//============================================================================
// Matrix reading functionality
//============================================================================

// Reading in a matrix from a stream
void ReadMat(std::ifstream& stream, cv::Mat &output_mat)
{
	// Read in the number of rows, columns and the data type
	int row,col,type;
	
	stream >> row >> col >> type;

	output_mat = cv::Mat(row, col, type);
	
	switch(output_mat.type())
	{
		case CV_64FC1: 
		{
			cv::MatIterator_<double> begin_it = output_mat.begin<double>();
			cv::MatIterator_<double> end_it = output_mat.end<double>();
			
			while(begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_32FC1:
		{
			cv::MatIterator_<float> begin_it = output_mat.begin<float>();
			cv::MatIterator_<float> end_it = output_mat.end<float>();

			while(begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_32SC1:
		{
			cv::MatIterator_<int> begin_it = output_mat.begin<int>();
			cv::MatIterator_<int> end_it = output_mat.end<int>();
			while(begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_8UC1:
		{
			cv::MatIterator_<uchar> begin_it = output_mat.begin<uchar>();
			cv::MatIterator_<uchar> end_it = output_mat.end<uchar>();
			while(begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		default:
			printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", __FILE__,__LINE__,output_mat.type()); abort();


	}
}

void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat)
{
	// Read in the number of rows, columns and the data type
	int row, col, type;
	
	stream.read ((char*)&row, 4);
	stream.read ((char*)&col, 4);
	stream.read ((char*)&type, 4);
	
	output_mat = cv::Mat(row, col, type);
	int size = output_mat.rows * output_mat.cols * output_mat.elemSize();
	stream.read((char *)output_mat.data, size);

}

// Skipping lines that start with # (together with empty lines)
void SkipComments(std::ifstream& stream)
{	
	while(stream.peek() == '#' || stream.peek() == '\n'|| stream.peek() == ' ' || stream.peek() == '\r')
	{
		std::string skipped;
		std::getline(stream, skipped);
	}
}

}
