#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

#include <unistd.h>

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::face;

#define CAM_WIDTH 640
#define CAM_HEIGHT 480

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

const std::string tensorflowConfigFile = "../data/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "../data/opencv_face_detector_uint8.pb";

void detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN)
{
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;

        cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }

}

/* drawLandmarks*/


#define COLOR Scalar(255, 200,0)


// drawPolyLine draws a poly line by joining 
// successive points between the start and end indices. 
void drawPolyline
(
  Mat &im,
  const vector<Point2f> &landmarks,
  const int start,
  const int end,
  bool isClosed = false
)
{
    // Gather all points between the start and end indices
    vector <Point> points;
    for (int i = start; i <= end; i++)
    {
        points.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
    }
    // Draw polylines. 
    polylines(im, points, isClosed, COLOR, 2, 16);
    
}


void drawLandmarks(Mat &im, vector<Point2f> &landmarks)
{
    // Draw face for the 68-point model.
    if (landmarks.size() == 68)
    {
      drawPolyline(im, landmarks, 0, 16);           // Jaw line
      drawPolyline(im, landmarks, 17, 21);          // Left eyebrow
      drawPolyline(im, landmarks, 22, 26);          // Right eyebrow
      drawPolyline(im, landmarks, 27, 30);          // Nose bridge
      drawPolyline(im, landmarks, 30, 35, true);    // Lower nose
      drawPolyline(im, landmarks, 36, 41, true);    // Left eye
      drawPolyline(im, landmarks, 42, 47, true);    // Right Eye
      drawPolyline(im, landmarks, 48, 59, true);    // Outer lip
      drawPolyline(im, landmarks, 60, 67, true);    // Inner lip
    }
    else 
    { // If the number of points is not 68, we do not know which 
      // points correspond to which facial features. So, we draw 
      // one dot per landamrk. 
      for(int i = 0; i < landmarks.size(); i++)
      {
        circle(im,landmarks[i],3, COLOR, FILLED);
      }
    }
    
}

//////////////////////////////////////////////////////////

int main( int argc, const char** argv )
{
  Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);

  VideoCapture source;
  if (argc == 1)
      source.open(0);
  else
      source.open(argv[1]);
  Mat frame;

  source.set(CAP_PROP_FRAME_WIDTH, CAM_WIDTH);
  source.set(CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT);

    // Create an instance of Facemark
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel("../data/lbfmodel.yaml");

  double tt_opencvDNN = 0;
  double fpsOpencvDNN = 0;

  const int fixed_delay = 250;
  while(1)
  {
      source >> frame;

      vector<Rect> faces;

      if(frame.empty())
          break;
      double t = cv::getTickCount();
      // face detection
      // TODO faces 에 찾을 얼굴 좌표를 넣도록 해야한다. 
      detectFaceOpenCVDNN ( net, frame );

      // landmark detection
      vector< vector<Point2f> > landmarks;      
       bool success = facemark->fit(frame,faces,landmarks);
      
      if(success)
      {
        // If successful, render the landmarks on the face
        for(int i = 0; i < landmarks.size(); i++)
        {
          drawLandmarks(frame, landmarks[i]);
        }
      }

      tt_opencvDNN = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
      int micro_sec_delay = static_cast<int>(tt_opencvDNN*1000);

      std::cout<<micro_sec_delay<<std::endl;
      usleep((fixed_delay - micro_sec_delay)*1000);

      tt_opencvDNN = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
      fpsOpencvDNN = 1/tt_opencvDNN;

      putText(frame, format("OpenCV DNN ; FPS = %.2f",fpsOpencvDNN), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
      imshow( "OpenCV - DNN Face Detection", frame );

      int k = waitKey(5);
      if(k == 27)
      {
        destroyAllWindows();
        break;
      }
    }
}
