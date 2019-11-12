//
//  CornerDescriptor.cpp
//  OpenCV-Template
//
//  Created by Benjamin Therien on 2019-03-22.
//  Copyright © 2019 Benjamin Therien. All rights reserved.
//

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <vector>
#include "Stitching.cpp"
//#inlcude <algorithm>

using namespace std;
using namespace cv;

/**
 @citation This code was strongly based off this:
     Title: Harris corner detector OpenCV 2.4.13.7
     Author: OpenCV Documenters
     Date: Copyright 2011 - 2014
     Availability: https://docs.opencv.org/2.4.13.7/doc/tutorials/features2d/trackingmotion/harris_detector/harris_detector.html
 
 @function getCorners
 @param img : the image we wish to extract corners from
 @param cornerScoreThreshold : threshold for the corner score
 @return keyPoints : vector containing the points whose value is greater than the cornerScoreThreshold
 */
static vector<KeyPoint> getCorners(Mat img, int cornerScoreThreshold, bool show)
{
    
    Mat dst, dst_norm, dst_norm_scaled, src_grey;
    dst = Mat::zeros( img.size(), CV_32FC1 );
    
    cvtColor( img, src_grey, CV_BGR2GRAY );
    
    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    
    /// Detecting corners
    cornerHarris(src_grey, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    
    /// Normalizing
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    
    
    
    vector<KeyPoint> keypoints;
    if(show)
    {
        Mat temp;
        img.copyTo(temp);
        /// Adding Keypoints
        for( int j = 0; j < dst_norm.rows ; j++ )
        {
            for( int i = 0; i < dst_norm.cols; i++ )
            {
                if( (int) dst_norm.at<float>(j,i) > cornerScoreThreshold )
                {
                    keypoints.push_back(*new KeyPoint( i, j, 1, -1, 0, 0, -1));
                    circle( temp, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
                }
            }
        }
        imshow("DetectCorners", temp);
        imwrite("/Users/BenjaminTherien/Desktop/COMP499_Project/1.png", temp);
        waitKey(0);
    }
    else
    {
        /// Adding Keypoints
        for( int j = 0; j < dst_norm.rows ; j++ )
        {
            for( int i = 0; i < dst_norm.cols; i++ )
            {
                if( (int) dst_norm.at<float>(j,i) > cornerScoreThreshold )
                {
                    keypoints.push_back(*new KeyPoint( i, j, 1, -1, 0, 0, -1));
                }
            }
        }
    }
    
    return keypoints;
}

/**
 @function stitchMultiple
 @param img_1 , img_2 : we want to find the homography between these images
 @param cornerThreshold : threshold for the corner score
 @param inlierDistance : distance used to determine if a point is an "inlier"
 @param iterations : the number of ransac iteration we wish to run
 @param angleThreshold : threshold on the distance from the median angle
 @param MDFThreshold : threshold on the distance from the median distance
 @param show : boolean to determine if point matching should be shown
 @return map containing the homography between img_1 and img_2 and its inverse
 */
static map<String, Mat> getHoms(Mat img_1, Mat img_2, double cornerThreshold, int iterations, double inlierDistance, double angleThreshold, double MDFThreshold,  bool show)
{
    
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    
    Mat siftMatches(img_1);
    
    //-- Step 1: Detect the keypoints:
    vector<KeyPoint> keypoints_1, keypoints_2;
    keypoints_1 = getCorners(img_1, cornerThreshold,show);
    keypoints_2 = getCorners(img_2, cornerThreshold,show);
    
    cout<<"corners Done with: i1 = "<<keypoints_1.size() << " and  I2 = " << keypoints_2.size()<<endl;
    
    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute( img_1, keypoints_1, descriptors_1 );
    f2d->compute( img_2, keypoints_2, descriptors_2 );
    
    
    //-- Step 3: Matching descriptor vectors using BFMatcher :
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches);
    
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, siftMatches);
    
    if(show == true)
    {
        imshow("matches", siftMatches);
        waitKey(0);
    }
    
    /*
     * Format KeyPoints to be Point2f
     **/
    
    vector<Point2f> matchPoints_1, matchPoints_2;
    
    for( int i = 0; i < matches.size(); i++)
    {
        matchPoints_1.push_back( keypoints_1[ matches[i].queryIdx ].pt);
        matchPoints_2.push_back( keypoints_2[ matches[i].trainIdx ].pt);
    }
    
    /*
     * Get the inliers of the homography computed from ransac
     * Then Display
     **/
    
    vector<Point2f>inliers1, inliers2;
    vector<Point2f>::iterator it1, it2;
    vector<DMatch> inlierMatches;
    keypoints_1.clear();
    keypoints_2.clear();
    //get homogaphy
    map<String, Mat> Homographies = ransac(matchPoints_1, matchPoints_2, inlierDistance, iterations, img_1.cols, MDFThreshold, angleThreshold);
    
    if(show == true)
    {
        Mat ransacMatches(img_1);
        getInliers(Homographies["Hom"], matchPoints_1, matchPoints_2, 10, inliers1, inliers2);
        int count=0;
        for(it1 = inliers1.begin(),it2=inliers2.begin(); it1< inliers1.end(); it1++,it2++,count++)
        {
            inlierMatches.push_back(DMatch(count, count, 0));
            keypoints_1.push_back(KeyPoint(it1->x,it1->y, 1, -1, 0, 0, -1));
            keypoints_2.push_back(KeyPoint(it2->x,it2->y, 1, -1, 0, 0, -1));
        }
        
        drawMatches(img_1, keypoints_1, img_2, keypoints_2, inlierMatches, ransacMatches);
        imshow("matches after ransac", ransacMatches);
        waitKey(0);
    }
    
    return Homographies;
}

/**
 @function pano
 @param img_1 , img_2 : we want to find the homography between these images
 @param cornerThreshold : threshold for the corner score
 @param inlierDistance : distance used to determine if a point is an "inlier"
 @param iterations : the number of ransac iteration we wish to run
 @param angleThreshold : threshold on the distance from the median angle
 @param MDFThreshold : threshold on the distance from the median distance
 @param show : boolean to determine if point matching should be shown
 @return map containing the homography between img_1 and img_2 and its inverse
 */
static Mat pano(Mat img_1, Mat img_2, double cornerThreshold, int iterations, double inlierDistance, double angleThreshold, double MDFThreshold,  bool show)
{
    
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    Mat siftMatches(img_1);
    
    //-- Step 1: Detect the keypoints:
    vector<KeyPoint> keypoints_1, keypoints_2;
    keypoints_1 = getCorners(img_1, cornerThreshold,show);
    keypoints_2 = getCorners(img_2, cornerThreshold,show);
    
    cout<<"corners Done with: i1 = "<<keypoints_1.size() << " and  I2 = "<<keypoints_2.size()<<endl;
    
    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute( img_1, keypoints_1, descriptors_1 );
    f2d->compute( img_2, keypoints_2, descriptors_2 );
    
    
    //-- Step 3: Matching descriptor vectors using BFMatcher :
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches);
    
    
    
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, siftMatches);
    
    if(show == true)
    {
        imshow("matches", siftMatches);
        waitKey(0);
    }
    
    /*
     * Corner Detection and Feature matching above
     * Homography computation and more below
     **/
    vector<Point2f> matchPoints_1, matchPoints_2;
    
    for( int i = 0; i < matches.size(); i++)
    {
        matchPoints_1.push_back( keypoints_1[ matches[i].queryIdx ].pt);
        matchPoints_2.push_back( keypoints_2[ matches[i].trainIdx ].pt);
    }
    
    /*
     * Get the inliers of the homography computed from ransac
     * Then Display
     **/
    vector<Point2f>inliers1, inliers2;
    vector<Point2f>::iterator it1, it2;
    vector<DMatch> inlierMatches;
    keypoints_1.clear();
    keypoints_2.clear();
    
    //get homogaphy
    map<String, Mat> Homographies = ransac(matchPoints_1, matchPoints_2, inlierDistance , iterations, img_1.cols, MDFThreshold, angleThreshold);
    
    
    if(show == true)
    {
        Mat ransacMatches(img_1);
        getInliers(Homographies["Hom"], matchPoints_1, matchPoints_2, 10, inliers1, inliers2);
        
        int count=0;
        for(it1 = inliers1.begin(),it2=inliers2.begin(); it1< inliers1.end(); it1++,it2++,count++)
        {
            inlierMatches.push_back(DMatch(count, count, 0));
            keypoints_1.push_back(KeyPoint(it1->x,it1->y, 1, -1, 0, 0, -1));
            keypoints_2.push_back(KeyPoint(it2->x,it2->y, 1, -1, 0, 0, -1));
        }
        
        drawMatches(img_1, keypoints_1, img_2, keypoints_2, inlierMatches, ransacMatches);
    
        imshow("matches", ransacMatches);
        waitKey(0);
    }
    
    
    return Stitch(img_1,img_2,Homographies);
    
}
