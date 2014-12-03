/*
 * Application.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Dawid Pajak
 */

#ifndef APPLICATION_H_
#define APPLICATION_H_

namespace dtslam
{

class Application
{
public:
    Application(void)
    {
    }

    ~Application(void)
    {
    }

    virtual void touchMove(int id, int x, int y)
    {
        (void)id;
        (void)x;
        (void)y;
    }

    virtual void touchDown(int id, int x, int y)
    {
        (void)id;
        (void)x;
        (void)y;
    }

    virtual void touchUp(int id, int x, int y)
    {
        (void)id;
        (void)x;
        (void)y;
    }

    virtual void touchCancel(int id, int x, int y)
    {
        (void)id;
        (void)x;
        (void)y;
    }

    virtual bool init(void)
    {
        return true;
    }

    virtual bool loop(void)
    {
        return true;
    }

    virtual void draw(void)
    {
    }

    virtual void exit(void)
    {
    }

    virtual void suspend(void)
    {
    }

    virtual void resume(void)
    {
    }

    //static managed_ptr<Application> AppInit(void);

protected:

};

}

#endif /* APPLICATION_H_ */
