/*
 * FeatureIndexer.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

 #ifndef FEATURE_INDEXER_H
#define FEATURE_INDEXER_H

#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <cassert>
#include <iostream>

#include "log.h"

namespace dtslam
{
////////////////////////////////////////////////////////////////////////
// Helper feature class for indexers that don't own the feature data
template<class TData>
class FeaturePointer
{
public:
	FeaturePointer(TData *data): mData(data) {}

	cv::Point2i getPosition() const {return mData->getPosition();}
	int getScore() const {return mData->getScore();}

	TData &operator *() const {return *mData;}
	TData *operator ->() const {return mData;}

	TData *getData() const {return mData;}

protected:
	TData *mData;
};

template <class TData>
FeaturePointer<TData> MakeFeaturePointer(TData *data) {return FeaturePointer<TData>(data);}

////////////////////////////////////////////////////////////////////////
// Helper feature class to use a cv::KeyPoint directly
class IndexedCvKeypoint: public cv::KeyPoint
{
public:
	IndexedCvKeypoint() {}

	IndexedCvKeypoint(const cv::KeyPoint &keypoint):
		cv::KeyPoint(keypoint)
	{}

	cv::Point2i getPosition() const {return pt;}
	int getScore() const { return (int)response; }
};

////////////////////////////////////////////////////////////////////////
// FeatureRowIndexer
template<class TData>
class FeatureRowIndexer
{
public:
	typedef typename std::vector<TData>::iterator iterator;

	iterator addFeature(const TData &feature);

	iterator getFirstInRect(const cv::Rect2i &rect);

	template <class T>
	void getFeaturesInRect(std::vector<T> &featuresInRect, const cv::Rect2i &rect);

	void getAllFeatures(std::vector<TData*> &features);

	size_t size() {return mFeatures.size();}
	void clear() {mFeatures.clear();}
	void erase(iterator it) {mFeatures.erase(it);}
	void pop_back() {mFeatures.pop_back();}

	TData &operator [](int i) {return mFeatures[i];}

	iterator begin() {return mFeatures.begin();}
	TData &back() {return mFeatures.back();}
	iterator end() {return mFeatures.end();}
	bool empty() {return mFeatures.empty();}

protected:
	std::vector<TData> mFeatures;

	iterator findStartOfRow(int y);

	void pushIntoVector(std::vector<TData*> &vec, const iterator &it) {vec.push_back(&(*it));}
	void pushIntoVector(std::vector<iterator> &vec, const iterator &it) {vec.push_back(it);}
};


////////////////////////////////////////////////////////////////////////
// FeatureGridIndexerIterator
template<class TData>
class FeatureGridIndexer;

template<class TData>
class FeatureGridIndexerIterator
{
protected:
	typename std::vector<FeatureRowIndexer<TData>>::iterator mTilesEnd;
	typename std::vector<FeatureRowIndexer<TData>>::iterator mTileIt;
	typename std::vector<TData>::iterator mFeatureIt;
public:
	friend class FeatureGridIndexer<TData>;

	FeatureGridIndexerIterator()
	{
		mTilesEnd = typename std::vector<FeatureRowIndexer<TData>>::iterator();
		mTileIt = typename std::vector<FeatureRowIndexer<TData>>::iterator();
		mFeatureIt = typename std::vector<TData>::iterator();
	}
	FeatureGridIndexerIterator(typename std::vector<FeatureRowIndexer<TData>>::iterator tilesBegin, typename std::vector<FeatureRowIndexer<TData>>::iterator tilesEnd):
		FeatureGridIndexerIterator()
	{
		mTilesEnd = tilesEnd;
		mTileIt = tilesBegin;
		if(mTileIt != mTilesEnd)
		{
			mFeatureIt = mTileIt->begin();
			findValidFeatureIt();
		}
		else
			mFeatureIt = typename std::vector<TData>::iterator();
	}
	FeatureGridIndexerIterator(typename std::vector<FeatureRowIndexer<TData>>::iterator tile, typename FeatureRowIndexer<TData>::iterator feature, typename std::vector<FeatureRowIndexer<TData>>::iterator tilesEnd)
	{
		mTilesEnd = tilesEnd;
		mTileIt = tile;
		mFeatureIt = feature;
		findValidFeatureIt();
	}

	void findValidFeatureIt()
	{
		while(mFeatureIt==mTileIt->end())
		{
			++mTileIt;
			if(mTileIt == mTilesEnd)
			{
				mFeatureIt = typename std::vector<TData>::iterator();
				break;
			}
			else
				mFeatureIt = mTileIt->begin();
		}
	}

	bool operator==(const FeatureGridIndexerIterator &other)
	{
		if (mTileIt == other.mTileIt)
		{
			if (mTileIt == mTilesEnd)
				return true;
			else
				return (mFeatureIt == other.mFeatureIt);
		}
		else
			return false;
	}
	bool operator!=(const FeatureGridIndexerIterator &other)
	{
		return !(*this == other);
	}

	FeatureGridIndexerIterator &operator++()
	{
		++mFeatureIt;
		findValidFeatureIt();
		return *this;
	}

	TData &operator *()
	{
		return *mFeatureIt;
	}

	TData *operator ->()
	{
		return &(*mFeatureIt);
	}
};

////////////////////////////////////////////////////////////////////////
// FeatureGridIndexer
/**
 * @brief Class to index 2D features efficiently. Sorts them into a grid for fast read access.
 */
template<class TData>
class FeatureGridIndexer
{
public:
	typedef FeatureGridIndexerIterator<TData> iterator;

	FeatureGridIndexer() {}
	FeatureGridIndexer(cv::Size2i imageSize, cv::Size2i tilePixelSize, int nonMaximaSize=0) {create(imageSize,tilePixelSize,nonMaximaSize);}

	void create(cv::Size2i imageSize, cv::Size2i tilePixelSize, int nonMaximaSize=0);

	FeatureGridIndexer<TData> applyNonMaximaSuppresion(int nonMaximaSize);
	
	template<class T>
	static FeatureGridIndexer<TData> ApplyNonMaximaSuppresion(const T &container, const cv::Size2i &imageSize, const cv::Size2i &tileSize, int nonMaximaSize);

	cv::Size2i getGridSize() const {return mGridSize;}
	cv::Size2i getImageSize() const {return mImageSize;}
	cv::Size2i getTilePixelSize() const {return mTilePixelSize;}
	int getTileCount() const {return mGridSize.width*mGridSize.height;}

	int getNonMaximaSize() const {return mNonMaximaSize;}
	void setNonMaximaSize(int value) {mNonMaximaSize = value;}

	std::vector<FeatureRowIndexer<TData>> &getTiles() {return mTiles;}

	const FeatureRowIndexer<TData> &getTile(int i) const {return mTiles[i];}
	FeatureRowIndexer<TData> &getTile(int i) {return mTiles[i];}
	const FeatureRowIndexer<TData> &getTile(int x, int y) const {return mTiles[mGridSize.width*y + x];}
	FeatureRowIndexer<TData> &getTile(int x, int y) {return mTiles[mGridSize.width*y + x];}
	FeatureRowIndexer<TData> &getTile(const cv::Point2i &tp) {return getTile(tp.x,tp.y);}

	TData *addFeature(const TData &feature);

	void getFeaturesInRect(std::vector<TData*> &featuresInRect, const cv::Rect2i &rect);

	void clear();
	void erase(iterator it) {it.mTileIt->erase(it.mFeatureIt);}

	size_t size();

	iterator begin() {return FeatureGridIndexerIterator<TData>(mTiles.begin(), mTiles.end());}
	iterator end() { return FeatureGridIndexerIterator<TData>(mTiles.end(), mTiles.end()); }

protected:
	cv::Size2i mGridSize;
	cv::Size2i mImageSize;
	cv::Size2i mTilePixelSize;

	std::vector<FeatureRowIndexer<TData>> mTiles;

	int mNonMaximaSize;

	cv::Point2i getTileCoord(const cv::Point2i &p) const {return getTileCoord(p.x,p.y);}
	cv::Point2i getTileCoord(int x, int y) const
	{
		cv::Point2i p(x / mTilePixelSize.width, y / mTilePixelSize.height);
		if(p.x<0)
			p.x=0;
		else if(p.x>=mGridSize.width)
			p.x = mGridSize.width-1;
		if(p.y<0)
			p.y=0;
		else if(p.y>=mGridSize.height)
			p.y = mGridSize.height-1;
		return p;
	}

	bool getFirstInRect(FeatureRowIndexer<TData> *&tile, typename FeatureRowIndexer<TData>::iterator &feature, const cv::Rect2i &rect);
};

/////////////////////////////////////////////////////////////////////////////////////////
// Template implementations
// FeatureRowIndexer
template<class TData>
typename FeatureRowIndexer<TData>::iterator FeatureRowIndexer<TData>::addFeature(const TData &feature)
{
	for(auto it=mFeatures.begin(); it!=mFeatures.end(); it++)
	{
		if(it->getPosition().y > feature.getPosition().y)

		{
			//Insert here
			return mFeatures.insert(it,feature);
		}
	}

	//Insert at end
	mFeatures.push_back(feature);
	return mFeatures.end()-1;
}

template<class TData>
typename FeatureRowIndexer<TData>::iterator FeatureRowIndexer<TData>::findStartOfRow(int y)
{
	//TODO: The features are sorted but we do this sequentially. Would be faster to use a binary search or something.
	for(auto it=mFeatures.begin(); it!=mFeatures.end(); it++)
	{
		if(it->getPosition().y >= y)
			return it;
	}
	return mFeatures.end();
}

template<class TData>
typename FeatureRowIndexer<TData>::iterator FeatureRowIndexer<TData>::getFirstInRect(const cv::Rect2i &rect)
{
	const cv::Point2i br = rect.br();
	iterator it = findStartOfRow(rect.y);

	while(it!=mFeatures.end())
	{
		if(it->getPosition().y >= br.y)
			break;
		if(it->getPosition().x >= rect.x && it->getPosition().x < br.x)
			return it;
		++it;
	}
	return mFeatures.end();
}

template <class TData>
template <class T>
void FeatureRowIndexer<TData>::getFeaturesInRect(std::vector<T> &featuresInRect, const cv::Rect2i &rect)
{
	const cv::Point2i br = rect.br();
	auto it = findStartOfRow(rect.y);

	while(it!=mFeatures.end())
	{
		if(it->getPosition().y >= br.y)
			break;
		if(it->getPosition().x >= rect.x && it->getPosition().x < br.x)
			pushIntoVector(featuresInRect, it);
		++it;
	}
}

template<class TData>
void FeatureRowIndexer<TData>::getAllFeatures(std::vector<TData*> &features)
{
	//features.insert(features.end(), mFeatures.begin(), mFeatures.end());
	for(auto it=mFeatures.begin(); it!=mFeatures.end(); it++)
		features.push_back(&(*it));
}

////////////////////////////////////////////////////////////////////////
// FeatureGridIndexer

template<class TData>
void FeatureGridIndexer<TData>::create(cv::Size2i imageSize, cv::Size2i tilePixelSize, int nonMaximaSize)
{
	mImageSize = imageSize;
	mTilePixelSize = tilePixelSize;

	mGridSize.width = (int)std::ceil(static_cast<float>(imageSize.width) / tilePixelSize.width);
	mGridSize.height = (int)std::ceil(static_cast<float>(imageSize.height) / tilePixelSize.height);

	mTiles.clear();
	mTiles.resize(mGridSize.area(), FeatureRowIndexer<TData>());

	mNonMaximaSize = nonMaximaSize;
}

template<class TData>
FeatureGridIndexer<TData> FeatureGridIndexer<TData>::applyNonMaximaSuppresion(int nonMaximaSize)
{
	FeatureGridIndexer<TData> newGrid;
	newGrid.create(mImageSize, mTilePixelSize, nonMaximaSize);
	for(auto it=begin(); it!=end(); ++it)
		newGrid.addFeature(*it);

	return newGrid;
}

template<class TData>
template<class T>
FeatureGridIndexer<TData> FeatureGridIndexer<TData>::ApplyNonMaximaSuppresion(const T &container, const cv::Size2i &imageSize, const cv::Size2i &tileSize, int nonMaximaSize)
{
	FeatureGridIndexer<TData> newGrid;
	newGrid.create(imageSize, tileSize, nonMaximaSize);
	for (auto it = std::begin(container); it != std::end(container); ++it)
		newGrid.addFeature(*it);
	
	return newGrid;
}


template<class TData>
TData *FeatureGridIndexer<TData>::addFeature(const TData &feature)
{
	assert(mTiles.size() != 0);

	if(mNonMaximaSize == 0)
	{
		FeatureRowIndexer<TData> &tile = getTile(getTileCoord(feature.getPosition()));
		return &*tile.addFeature(feature);
	}
	else
	{
		bool add=true;
		const cv::Rect2i rect((int)feature.getPosition().x - mNonMaximaSize, (int)feature.getPosition().y - mNonMaximaSize, 2 * mNonMaximaSize, 2 * mNonMaximaSize);

		const cv::Point2i br = rect.br();
		const cv::Point2i tp = getTileCoord(rect.x, rect.y);
		const cv::Point2i tbr = getTileCoord(br.x, br.y);

		for(int j=tp.y; j<=tbr.y && add; j++)
		{
			for(int i=tp.x; i<=tbr.x && add; i++)
			{
				FeatureRowIndexer<TData> &tile = getTile(i,j);
				if(j>tp.y && j<tbr.y && i>tp.x && i<tbr.x)
				{
					while(!tile.empty())
					{
						if(tile.back().getScore() >= feature.getScore())
						{
							add = false;
							break;
						}
						else
						{
							tile.pop_back();
						}
					}
				}
				else
				{
					auto prev = tile.getFirstInRect(rect);
					while(prev != tile.end())
					{
						if(prev->getScore()>= feature.getScore())
						{
							add = false;
							break;
						}
						else
						{
							tile.erase(prev);
						}
						prev = tile.getFirstInRect(rect);
					}
				}
			}
		}

//		FeatureRowIndexer::iterator prev;
//
//		if(getFirstInRect(tile,prev,rect))
//		{
//			if(prev->response >= feature.response)
//				add = false;
//			else
//			{
//				cv::Point2f lastpt = prev->pt;
//
//				tile->erase(prev);
//				if(getFirstInRect(tile,prev,rect))
//				{
//					std::stringstream ss;
//					ss << "Last:(" << lastpt.x << "," << lastpt.y << "), second:(" << prev->pt.x << "," << prev->pt.y << ")" << std::endl;
//					LOG(ss.str().c_str());
//				}
//			}
//		}

		if(add)
		{
			FeatureRowIndexer<TData> &tile = getTile(getTileCoord(feature.getPosition()));
			return &*tile.addFeature(feature);
		}
		return NULL;
	}
}

template<class TData>
void FeatureGridIndexer<TData>::clear()
{
	for(auto it=mTiles.begin(); it!=mTiles.end(); ++it)
		it->clear();
}

template<class TData>
bool FeatureGridIndexer<TData>::getFirstInRect(FeatureRowIndexer<TData> *&tile, typename FeatureRowIndexer<TData>::iterator &feature, const cv::Rect2i &rect)
{
	const cv::Point2i br = rect.br();
	const cv::Point2i tp = getTileCoord(rect.x, rect.y);
	const cv::Point2i tbr = getTileCoord(br.x, br.y);

	for(int j=tp.y; j<=tbr.y; j++)
	{
		for(int i=tp.x; i<=tbr.x; i++)
		{
			FeatureRowIndexer<TData> &tile_i = getTile(i,j);
			if(j>tp.y && j<tbr.y && i>tp.x && i<tbr.x)
			{
				if(tile_i.begin() != tile_i.end())
				{
					tile = &tile_i;
					feature = tile_i.begin();
					return true;
				}
			}
			else
			{
				feature = tile_i.getFirstInRect(rect);
				if(feature != tile_i.end())
				{
					tile = &tile_i;
					return true;
				}
			}
		}
	}

	return false;
}

template<class TData>
void FeatureGridIndexer<TData>::getFeaturesInRect(std::vector<TData*> &featuresInRect, const cv::Rect2i &rect)
{
	const cv::Point2i br = rect.br();
	const cv::Point2i tp = getTileCoord(rect.x, rect.y);
	const cv::Point2i tbr = getTileCoord(br.x, br.y);

	for(int j=tp.y; j<=tbr.y; j++)
	{
		for(int i=tp.x; i<=tbr.x; i++)
		{
			FeatureRowIndexer<TData> &tile = getTile(i,j);
			if(j>tp.y && j<tbr.y && i>tp.x && i<tbr.x)
				tile.getAllFeatures(featuresInRect);
			else
				tile.getFeaturesInRect(featuresInRect, rect);
		}
	}
}

template<class TData>
size_t FeatureGridIndexer<TData>::size()
{
	size_t s=0;
	for(auto it=mTiles.begin(); it!=mTiles.end(); ++it)
		s+=it->size();
	return s;
}

}

#endif
