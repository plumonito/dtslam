/*
 * ViewportTiler.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */
 
#ifndef VIEWPORT_TILER_H_
#define VIEWPORT_TILER_H_

#include <opencv2/core.hpp>
#include <memory>
#include <cassert>

namespace dtslam
{

/**
 * @brief Class for tiling the output opengl contexts
 */
class ViewportTiler
{
public:
	ViewportTiler();

    int getRows() const {return mRows;}
    int getCols() const {return mCols;}

    const cv::Rect2i &getViewportArea() const {return mViewportArea;}

    void configDevice(const cv::Rect2i &viewportArea, int rows, int cols);
    void configDevice(const cv::Rect2i &viewportArea, int cols) {configDevice(viewportArea, 1, cols);}

    void addTile(int row, int col, float aspectRatio, int rowSpan, int colSpan);
    void addTile(int row, int col, float aspectRatio) {addTile(row, col, -1.0f, 1, 1);}
    void addTile(int row, int col) {addTile(row, col, -1.0f);}
    void addTile(int col) {addTile(1, col);}

    void fillTiles();
    int getTileCount() {return (int)mTiles.size();}

    void setActiveTile(int index)
    {
    	assert(index>=0 && index<(int)mTiles.size());
    	setActiveTile(mTiles[index].get());
    }

    void setFullScreen()
    {
    	setActiveTile(&mFullScreenTile);
    }

    const cv::Matx44f &getMVP() const {return getMVP(mActiveTile);}
    const cv::Matx44f &getMVP(int tileIndex) const {assert(tileIndex<(int)mTiles.size()); return mTiles[tileIndex]->mMvp;}
    void resetMVP() {resetMVP(mActiveTile);}
    void resetMVP(int tileIndex) {assert(tileIndex<(int)mTiles.size()); resetMVP(mTiles[tileIndex].get());}
    void multiplyMVP(const cv::Matx44f &mvp) {multiplyMVP(mActiveTile,mvp);}
    void multiplyMVP(int tileIndex, const cv::Matx44f &mvp) {assert(tileIndex<(int)mTiles.size()); multiplyMVP(mTiles[tileIndex].get(),mvp);}
    void setImageMVP(const cv::Size2i &imageSize) {setImageMVP(mActiveTile,imageSize);}
    void setImageMVP(int tileIndex, const cv::Size2i &imageSize) {assert(tileIndex<(int)mTiles.size()); setImageMVP(mTiles[tileIndex].get(),imageSize);}

    static cv::Matx44f GetImageSpaceMvp(float viewportAspect, const cv::Size2i &imageSize);
    static cv::Matx44f GetImageSpaceMvp(const cv::Size2i &viewportSize, const cv::Size2i &imageSize)
    {
    	return GetImageSpaceMvp(static_cast<float>(viewportSize.width)/viewportSize.height, imageSize);
    }

    void screenToVertex(const cv::Point2f &screenPoint, int &tileIdx, cv::Vec4f &vertex) const;

protected:
    /**
     * @brief Class for the tiles
     */
    class ViewportTileInfo
    {
    public:
        cv::Rect2i mViewportArea;
        float mAspectRatio; //The aspect ratio of the logical viewport after the mMvp transform.
        cv::Matx44f mMvp;
    };

    int mRows;
    int mCols;

    cv::Rect2i mViewportArea;
    cv::Size2i mCellSize;

    ViewportTileInfo mFullScreenTile; //Special tile that is used for full screen support

    std::vector<std::unique_ptr<ViewportTileInfo>> mTiles; //List of tiles (owns and disposes of tile objects)
    std::vector<ViewportTileInfo*> mTileMatrix; //Row-major matrix (tiles spanning multiple cells appear several times)
    ViewportTileInfo *mActiveTile;

    const cv::Matx44f &getMVP(ViewportTileInfo *tile) const {return tile->mMvp;}
    void resetMVP(ViewportTileInfo *tile);
    void multiplyMVP(ViewportTileInfo *tile, const cv::Matx44f &mvp)
    {
    	tile->mMvp = tile->mMvp*mvp;
    }
    void setImageMVP(ViewportTileInfo *tile, const cv::Size2i &imageSize);

    void setActiveTile(ViewportTileInfo *tile);
    cv::Matx44f createAspectMVP(float viewportAspect, float newAspect);
};

}

#endif
