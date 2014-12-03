/*
 * BaseWindow.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef BASEWINDOW_H_
#define BASEWINDOW_H_

#include <string>
#include <opencv2/core.hpp>
#include <cassert>

#undef ERROR //From miniglog
#include <GL/glew.h>
#include <GL/freeglut.h>
#undef ERROR //From miniglog

#include <dtslam/log.h>
#include "../UserInterfaceInfo.h"
#include "../ViewportTiler.h"
#include "../TextureHelper.h"

namespace dtslam
{

class SlamDriver;
class ImageDataSource;
class DTSlamShaders;
class SlamSystem;

class BaseWindow;

template<class TBase>
class KeyBindingHandler
{
public:
	typedef void(TBase::*BindingFunc)(bool isSpecial, unsigned char key);
	typedef void(TBase::*SimpleBindingFunc)();

	KeyBindingHandler(TBase *target): mTarget(target) {}

	void clear() {mKeyBindings.clear();}

	void addBinding(bool isSpecial_, unsigned char key_, BindingFunc funcDown_, BindingFunc funcUp_,	std::string description_);
	void addBinding(bool isSpecial_, unsigned char key_, BindingFunc funcDown_, std::string description_);
	void addBinding(bool isSpecial_, unsigned char key_, SimpleBindingFunc funcDown_, SimpleBindingFunc funcUp_,	std::string description_);
	void addBinding(bool isSpecial_, unsigned char key_, SimpleBindingFunc funcDown_, std::string description_);
	bool dispatchKeyDown(bool isSpecial, unsigned char key);
	bool dispatchKeyUp(bool isSpecial, unsigned char key);

	void showHelp() const;

protected:
	struct Binding
	{
	public:
		Binding(bool isSpecial_, unsigned char key_, BindingFunc funcDown_, BindingFunc funcUp_,	std::string description_):
			isSpecial(isSpecial_),
			key(key_),
			isSimple(false),
			funcDown(funcDown_),
			funcUp(funcUp_),
			funcSimpleDown(NULL),
			funcSimpleUp(NULL),
			description(description_)
		{}
		Binding(bool isSpecial_, unsigned char key_, BindingFunc funcDown_, std::string description_):
			isSpecial(isSpecial_),
			key(key_),
			isSimple(false),
			funcDown(funcDown_),
			funcUp(NULL),
			funcSimpleDown(NULL),
			funcSimpleUp(NULL),
			description(description_)
		{}

		Binding(bool isSpecial_, unsigned char key_, SimpleBindingFunc funcDown_, SimpleBindingFunc funcUp_,	std::string description_):
			isSpecial(isSpecial_),
			key(key_),
			isSimple(true),
			funcDown(NULL),
			funcUp(NULL),
			funcSimpleDown(funcDown_),
			funcSimpleUp(funcUp_),
			description(description_)
		{}
		Binding(bool isSpecial_, unsigned char key_, SimpleBindingFunc funcDown_, std::string description_):
			isSpecial(isSpecial_),
			key(key_),
			isSimple(true),
			funcDown(NULL),
			funcUp(NULL),
			funcSimpleDown(funcDown_),
			funcSimpleUp(NULL),
			description(description_)
		{}

		bool isSpecial;
		unsigned char key;

		bool isSimple;
		BindingFunc funcDown;
		BindingFunc funcUp;
		SimpleBindingFunc funcSimpleDown;
		SimpleBindingFunc funcSimpleUp;

		bool hasUpFunc() const {return isSimple ? (funcSimpleUp!=NULL) : (funcUp!=NULL);}
		bool hasDownFunc() const {return isSimple ? (funcSimpleDown!=NULL) : (funcDown!=NULL);}

		std::string description;
	};

	TBase *mTarget;
    std::vector<Binding> mKeyBindings;

    static void LogKeyName(bool isSpecial, unsigned char key);
};

class BaseWindow
{
public:
	BaseWindow(std::string name): mIsInitialized(false), mName(name), mKeyBindings(this) {}
	virtual ~BaseWindow() {}

	bool isInitialized() const {return mIsInitialized;}
	void requireInit() {mIsInitialized=false;}
	const std::string &getName() const {return mName;}

	virtual bool init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize);

	virtual void showHelp() const;

	void setCurrentImageTexture(int target, int id)
	{
	    mCurrentImageTextureTarget = target;
	    mCurrentImageTextureId = id;
	}

    void keyDown(bool isSpecial, unsigned char key);
    void keyUp(bool isSpecial, unsigned char key);

    static const int kMouseLeftButton=0x0000;
	static const int kMouseMiddleButton=0x0001;
	static const int kMouseRightButton=0x0002;
	static const int kMouseScrollUp=0x0003;
	static const int kMouseScrollDown=0x0004;

    virtual void touchDown(int id, int x, int y) {}
	virtual void touchMove(int x, int y) {}
    virtual void touchUp(int id, int x, int y) {}

    virtual void resize() {}
    virtual void updateState() {}
    virtual void draw() {}

protected:
	bool mIsInitialized;
	SlamDriver *mApp;
    DTSlamShaders *mShaders;
    SlamSystem *mSlam;
    cv::Size mImageSize;

    std::string mName;
    KeyBindingHandler<BaseWindow> mKeyBindings;

    int mCurrentImageTextureTarget;
    int mCurrentImageTextureId;
};

////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
template<class TBase>
void KeyBindingHandler<TBase>::addBinding(bool isSpecial_, unsigned char key_, BindingFunc funcDown_, BindingFunc funcUp_,	std::string description_)
{
	mKeyBindings.push_back(Binding(isSpecial_, key_, funcDown_, funcUp_,	description_));
}
template<class TBase>
void KeyBindingHandler<TBase>::addBinding(bool isSpecial_, unsigned char key_, BindingFunc funcDown_, std::string description_)
{
	mKeyBindings.push_back(Binding(isSpecial_, key_, funcDown_, description_));
}
template<class TBase>
void KeyBindingHandler<TBase>::addBinding(bool isSpecial_, unsigned char key_, SimpleBindingFunc funcDown_, SimpleBindingFunc funcUp_,	std::string description_)
{
	mKeyBindings.push_back(Binding(isSpecial_, key_, funcDown_, funcUp_,	description_));
}
template<class TBase>
void KeyBindingHandler<TBase>::addBinding(bool isSpecial_, unsigned char key_, SimpleBindingFunc funcDown_, std::string description_)
{
	mKeyBindings.push_back(Binding(isSpecial_, key_, funcDown_, description_));
}

template<class TBase>
bool KeyBindingHandler<TBase>::dispatchKeyDown(bool isSpecial, unsigned char key)
{
	for(auto it=mKeyBindings.begin(), end=mKeyBindings.end(); it!=end; ++it)
	{
		Binding &bind = *it;
		if(bind.isSpecial == isSpecial && bind.key == key)
		{
			if(bind.hasDownFunc())
			{
				if(bind.isSimple)
					(mTarget->*bind.funcSimpleDown)();
				else
					(mTarget->*bind.funcDown)(isSpecial, key);
				return true;
			}
		}
	}
	return false;
}

template<class TBase>
bool KeyBindingHandler<TBase>::dispatchKeyUp(bool isSpecial, unsigned char key)
{
	for(auto it=mKeyBindings.begin(), end=mKeyBindings.end(); it!=end; ++it)
	{
		Binding &bind = *it;
		if(bind.isSpecial == isSpecial && bind.key == key)
		{
			if(bind.hasUpFunc())
			{
				if(bind.isSimple)
					(mTarget->*bind.funcSimpleUp)();
				else
					(mTarget->*bind.funcUp)(isSpecial, key);
				return true;
			}
		}
	}
	return false;
}

template<class TBase>
void KeyBindingHandler<TBase>::showHelp() const
{
	if(mKeyBindings.empty())
		DTSLAM_LOG << "No keyboard bindings.\n";
	else
	{
		DTSLAM_LOG << "Keyboard bindings:\n";
		for(auto it=mKeyBindings.begin(), end=mKeyBindings.end(); it!=end; ++it)
		{
			const Binding &bind = *it;

			DTSLAM_LOG << " - Key ";
			LogKeyName(bind.isSpecial, bind.key);
			DTSLAM_LOG << ": " << bind.description << "\n";
		}
	}
}

template<class TBase>
void KeyBindingHandler<TBase>::LogKeyName(bool isSpecial, unsigned char key)
{
	if(!isSpecial && key >= 32)
		DTSLAM_LOG << "'" << key << "'";
	else if(isSpecial && (key>=GLUT_KEY_F1 && key<=GLUT_KEY_F12))
	{
		DTSLAM_LOG << "'F" << (int)(key-GLUT_KEY_F1+1) << "'";
	}
	else if(!isSpecial && key == 27)
		DTSLAM_LOG << "'Esc'";
	else
		DTSLAM_LOG << (int)key;
}

} /* namespace dtslam */

#endif /* BASEWINDOW_H_ */
