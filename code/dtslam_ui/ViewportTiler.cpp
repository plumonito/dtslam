/*
 * ViewportTiler.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "ViewportTiler.h"
#include <cassert>
#include "GL/glew.h"

namespace dtslam
{

ViewportTiler::ViewportTiler()
        : mRows(0), mCols(0)
{

}

void ViewportTiler::configDevice(const cv::Rect2i &viewportArea, int rows, int cols)
{
	assert(rows>0 && cols>0);
	mRows = rows;
	mCols = cols;

	mViewportArea = viewportArea;
	mCellSize = cv::Size2i(viewportArea.width/cols, viewportArea.height/rows);

	mFullScreenTile.mAspectRatio = (float)mViewportArea.width / mViewportArea.height;
	mFullScreenTile.mViewportArea = mViewportArea;
	setImageMVP(&mFullScreenTile, mViewportArea.size());

	mTiles.clear();
	mTileMatrix.clear();
	mTileMatrix.resize(rows*cols, NULL);
}

void ViewportTiler::addTile(int row, int col, float aspectRatio, int rowSpan, int colSpan)
{
	ViewportTileInfo *tile = new ViewportTileInfo();
	mTiles.push_back(std::unique_ptr<ViewportTileInfo>(tile));

	tile->mViewportArea.x = mViewportArea.x + col*mCellSize.width;
	tile->mViewportArea.y = mViewportArea.y + row*mCellSize.height;
	tile->mViewportArea.width = mCellSize.width*colSpan;
	tile->mViewportArea.height = mCellSize.height*rowSpan;

	tile->mAspectRatio = aspectRatio;
	resetMVP(tile);

	int endRow=row+rowSpan;
	int endCol=col+colSpan;
	for(int j=row; j<endRow; j++)
	{
		for(int i=col; i<endCol; i++)
		{
			ViewportTileInfo *&cell = mTileMatrix[j*mCols + i];
			assert(cell==NULL);
			cell = tile;
		}
	}
}

void ViewportTiler::resetMVP(ViewportTileInfo *tile)
{
	//Configure mvp matrix to correct aspect ratio
	float viewportAspect = static_cast<float>(tile->mViewportArea.width) / tile->mViewportArea.height;
	if(tile->mAspectRatio == -1.0f)
	{
		tile->mAspectRatio = viewportAspect;
		tile->mMvp = cv::Matx44f::eye();
	}
	else
	{
		tile->mMvp = createAspectMVP(viewportAspect, tile->mAspectRatio);
	}
}

cv::Matx44f ViewportTiler::createAspectMVP(float viewportAspect, float newAspect)
{
	if(viewportAspect > newAspect)
	{
		//Shrink in x
		return cv::Matx44f(newAspect/viewportAspect,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
	}
	else
	{
		//Shrink in y
		return cv::Matx44f(1,0,0,0, 0,viewportAspect/newAspect,0,0, 0,0,1,0, 0,0,0,1);
	}
}

void ViewportTiler::setImageMVP(ViewportTileInfo *tile, const cv::Size2i &imageSize)
{
	float viewportAspect = static_cast<float>(tile->mViewportArea.width) / tile->mViewportArea.height;
	tile->mMvp = GetImageSpaceMvp(viewportAspect, imageSize);
}


cv::Matx44f ViewportTiler::GetImageSpaceMvp(float viewportAspect, const cv::Size2i &imageSize)
{
	float imageAspect = static_cast<float>(imageSize.width) / imageSize.height;
	cv::Size2f paddedSize = imageSize;
    if (viewportAspect > imageAspect)
    {
        // Viewport is wider than image
        // Use entire viewport height and add padding on x
    	paddedSize.width += viewportAspect * imageSize.height - imageSize.width;
    }
    else
    {
        // Viewport is taller than rectangle
        // Use entire viewport width and add padding on y
    	paddedSize.height += imageSize.width / viewportAspect - imageSize.height;
    }

    float zNear = 0.001f;
    float zFar = 100.0f;

    const float sx = (2.0f) / (paddedSize.width);
    const float tx = -((imageSize.width) / 2.0f) * sx;
    //const float tx = -1;
    const float sy = -(2.0f) / (paddedSize.height);
    const float ty = -((imageSize.height) / 2.0f) * sy;
    //const float ty = 1;
    const float sz = -2.0f / (zFar - zNear);
    const float tz  = -2.0f / (zFar + zNear);

    return cv::Matx44f(sx,0,tx,0, 0,sy,ty,0, 0,0,sz,tz, 0,0,1,0);
}

void ViewportTiler::fillTiles()
{
	for(int j=0; j<mRows; j++)
	{
		for(int i=0; i<mCols; i++)
		{
			ViewportTileInfo *&cell = mTileMatrix[j*mCols + i];
			if(cell==NULL)
				addTile(j,i);
		}
	}
}

void ViewportTiler::setActiveTile(ViewportTileInfo *tile)
{
	mActiveTile = tile;
	glViewport(mActiveTile->mViewportArea.x, mActiveTile->mViewportArea.y, mActiveTile->mViewportArea.width, mActiveTile->mViewportArea.height);
}

void ViewportTiler::screenToVertex(const cv::Point2f &screenPoint, int &tileIdx, cv::Vec4f &vertex) const
{
	int tileX = (int)(screenPoint.x / mCellSize.width);
	int tileY = (int)(screenPoint.y / mCellSize.height);

	if(tileX < 0 || tileX >= mCols || tileY < 0 || tileY >= mRows)
	{
		tileIdx = -1;
		return;
	}

	tileIdx = tileY*mCols + tileX;
	const auto &tile = *mTiles[tileIdx];
	cv::Matx44f mvpInv = tile.mMvp.inv();
	vertex = mvpInv * cv::Vec4f(2*(screenPoint.x-tile.mViewportArea.x)/tile.mViewportArea.width-1, 1-2*(screenPoint.y-tile.mViewportArea.y)/tile.mViewportArea.height, 1, 1);
}

}
