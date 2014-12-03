/*
 * main.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include <memory>
#include <iostream>
#include "dtslam_ui/SlamDriver.h"

#include <dtslam_ui/UserInterfaceInfo.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#undef GFLAGS_DLL_DEFINE_FLAG
#define GFLAGS_DLL_DEFINE_FLAG
#include <gflags/gflags.h>

namespace dtslam
{
DEFINE_int32(WindowWidth, 640, "Initial width of the window.");
DEFINE_int32(WindowHeight, 480, "Initial height of the window.");
}


dtslam::SlamDriver *gApp = NULL;

int gWindowId;

void changeSize(int w, int h)
{
	dtslam::UserInterfaceInfo::Instance().setScreenSize(cv::Size(w,h));
	gApp->resize();
}


void renderScene(void)
{
	if(gApp->getFinished())
	{
		glutDestroyWindow(gWindowId);
		exit(0);
	}

	gApp->draw();
	glutSwapBuffers();
}

void pressKey(unsigned char key, int x, int y)
{
	dtslam::UserInterfaceInfo::Instance().setKeyState(key, true);
	gApp->keyDown(false, key);
}

void releaseKey(unsigned char key, int x, int y)
{
	dtslam::UserInterfaceInfo::Instance().setKeyState(key, false);
	gApp->keyUp(false, key);
}

void pressSpecial(int key, int x, int y)
{
	dtslam::UserInterfaceInfo::Instance().setSpecialKeyState(key, true);
	gApp->keyDown(true, key);
}

void releaseSpecial(int key, int x, int y)
{
	dtslam::UserInterfaceInfo::Instance().setSpecialKeyState(key, false);
	gApp->keyUp(true, key);
}

void mouseEvent(int id, int state, int x, int y)
{
	if(state==GLUT_DOWN)
		gApp->touchDown(id, x, y);
	else if(state == GLUT_UP)
		gApp->touchUp(id, x, y);
}

void mouseMoveEvent(int x, int y)
{
	gApp->touchMove(x, y);
}

int main(int argc, char**argv)
{
	google::ParseCommandLineFlags(&argc, &argv, true);

	cv::Size initialSize(dtslam::FLAGS_WindowWidth, dtslam::FLAGS_WindowHeight);

	//init GLUT and create window
	glutInit(&argc, argv );
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA);
	//glutInitWindowPosition(900,10);
	glutInitWindowSize(initialSize.width,initialSize.height);
	dtslam::UserInterfaceInfo::Instance().setScreenSize(initialSize);
	gWindowId = glutCreateWindow("dslam");

	//Glew
	if(glewInit() != GLEW_OK)
	{
		DTSLAM_LOG << "Error initializing GLEW!\n";
		return 0;
	}

	// register callbacks
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutIdleFunc(renderScene);

	glutIgnoreKeyRepeat(0);
	glutKeyboardFunc(pressKey);
	glutKeyboardUpFunc(releaseKey);
	glutSpecialFunc(pressSpecial);
	glutSpecialUpFunc(releaseSpecial);
	glutMouseFunc(mouseEvent);
	glutMotionFunc(mouseMoveEvent);

	gApp = new dtslam::SlamDriver();

	if (!gApp->init())
	{
		delete gApp;
		return 1;
	}

	// enter GLUT event processing cycle
	glutMainLoop();

	delete gApp;

	return 0;
}
