/*
 * SequenceDataSource.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef SEQUENCEDATASOURCE_H_
#define SEQUENCEDATASOURCE_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <GL/glew.h>
#include "ImageDataSource.h"

namespace dtslam {

class SequenceDataSource: public ImageDataSource
{
public:
	SequenceDataSource(void);
    ~SequenceDataSource(void);

    bool open(const std::string &sequenceFormat, int startIdx);
    void close(void);

    void dropFrames(int count);
    bool update(void);

private:
    std::string mSequenceFormat;
    int mCurrentFrameIdx;

    cv::Mat readImage(int idx);
};

} /* namespace dtslam */

#endif /* SEQUENCEDATASOURCE_H_ */
