#include <limits.h>
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>
#include "FeatureIndexer.h"
#include <gl/gl.h>
#include <gl/glu.h>

void drawTestGl(void *)
{
	glClearColor(0,0,0,0);
	   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);     // Clear The Screen And The Depth Buffer
	    glLoadIdentity();

	    glTranslatef(-1.5f,0.0f,-6.0f);
	    //glColor3ub(255,0,255);

	glBegin(GL_TRIANGLES);
	glVertex3f(-0.5f, -0.5f, 1.0f);
	glVertex3f(-0.5f, 0.5f, 1.0f);
	glVertex3f(0.5f, 0.5f, 1.0f);
	glEnd();

	glBegin(GL_TRIANGLES);                      // Drawing Using Triangles
	    glVertex3f( 0.0f, 1.0f, 0.0f);              // Top
	    glVertex3f(-1.0f,-1.0f, 0.0f);              // Bottom Left
	    glVertex3f( 1.0f,-1.0f, 0.0f);              // Bottom Right
	glEnd();                            // Finished Drawing The Triangle

}

TEST(ZNCC, OpenGL)
{
	std::string windowName("OpenGL test");
	cv::namedWindow(windowName, cv::WINDOW_OPENGL + cv::WINDOW_NORMAL);
	cv::resizeWindow(windowName, 800, 600);

	glViewport(0,0,800,600);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f,(GLfloat)800/(GLfloat)600,0.1f,100.0f);
	glMatrixMode(GL_MODELVIEW_MATRIX);
	glLoadIdentity();

	glShadeModel(GL_SMOOTH);
	glDisable(GL_DEPTH_TEST);

	cv::setOpenGlDrawCallback(windowName, drawTestGl, NULL);
	cv::waitKey(5000);
}

TEST(ZNCC, Rotation)
{
	std::string windowName("Rotation");
	std::string datapath("D:/code/dslam/datasets");
	cv::VideoCapture cap(datapath + "/rotation.mp4");

	EXPECT_TRUE(cap.isOpened());


	cv::Mat frameRgb;
	int frameIdx=-1;
	while(cap.read(frameRgb))
	{
		std::cout << "Frame #" << ++frameIdx << std::endl;

		cv::Mat1b temp;
		cv::Mat1b frameGray;
		cv::cvtColor(frameRgb, frameGray, cv::COLOR_RGB2GRAY);

		cv::pyrDown(frameGray,temp);
		//cv::pyrDown(temp,frameGray);
		frameGray = temp;

		std::vector<cv::KeyPoint> keypoints;
		cv::FAST(frameGray, keypoints, 30, true);
		nvslam::FeatureGridIndexer<cv::KeyPoint> featureIndexer(cv::Size2i(frameGray.cols, frameGray.rows), cv::Size2i(40,40));
		for(auto it=keypoints.begin(); it!=keypoints.end(); it++)
			featureIndexer.addFeature(nvslam::IndexedFeature<cv::KeyPoint>(it->pt, it->response, *it));
		//for(auto it=featureIndexer.begin(); it!=featureIndexer.end(); ++it)
		//	std::cout << "(" << it->pt.x << "," << it->pt.y << ")" << std::endl;


		cv::Mat3b frameDraw(frameGray.rows, frameGray.cols);
		int fromTo[] = {0,0,0,1,0,2};
		cv::mixChannels(&frameGray, 1, &frameDraw, 1, fromTo, 3);

		//for(auto it=keypoints.begin(); it!=keypoints.end(); it++)
			cv::drawKeypoints(frameDraw,keypoints,frameDraw,cv::Scalar(0,255,0),cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

		cv::imshow(windowName, frameDraw);
		if(cv::waitKey(1) != -1)
			break;
	}
}
