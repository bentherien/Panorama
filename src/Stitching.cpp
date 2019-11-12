//
//  Stitching.cpp
//  OpenCV-Template
//
//  Created by Benjamin Therien on 2019-03-22.
//  Copyright © 2019 Benjamin Therien. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <vector>
#include "ransac.cpp"

using namespace std;
using namespace cv;

/**
 * @struct StitchedInfo
 * Use --> Container for a stitched image, and a point representing the displacement of the central image from
 */
struct StitchedInfo
{
    Mat StitchedImage;
    Point2f norm;
};
/**
 * @struct toStitch
 * Use --> stitching many images together, this is used to hold an image and its homography to the center images plane
 */
struct toStitch
{
    Mat image, HomToCenter;
    toStitch(Mat i, Mat h)
    {
        image = i;
        HomToCenter = h;
    }
};
/**
 @function getCornersVec
 @param Mat image : an image
 @return a vector containing coordinate points of each corner of the image
 */
static vector<Point2f> getCornersVec(Mat image)
{
    vector<Point2f> corners;
    corners.push_back(Point2f(0,0));
    corners.push_back(Point2f(image.cols,0));
    corners.push_back(Point2f(0,image.rows));
    corners.push_back(Point2f(image.cols,image.rows));
    return corners;
}

/**
 @function getProjectedCornersVec
 @param cornerInterest : the image, Mat H : the homography we want to warp the image with
 @return a vector containing coordinates of projected corners of the imge
 */
static vector<Point2f> getProjectedCornersVec(Mat cornerInterest, Mat H)
{
    vector<Point2f> corners = getCornersVec(cornerInterest), projectedCorners;
    vector<Point2f>::iterator it;
    
    for(it= corners.begin(); it<corners.end(); it++)
    {
        projectedCorners.push_back(project(it->x,it->y, H));
    }
    return projectedCorners;
}
/**
 @function getCornersVec
 @param image1 & 2 : images we wish to stitch together, HomInv : the homography from image 2 to 1
 @return a map containing 2 points, the maximum coordinates and minimum coordinates of the projected images
 */
static map<String, Point2f> getCornersVec(Mat image1,  Mat image2, Mat HomInv)
{
    vector<Point2f> image2Corners = getProjectedCornersVec(image2, HomInv);
    vector<Point2f> image1Corners = getCornersVec(image1);
    vector<Point2f>::iterator it1 ,it2;
    int maxX=0, minX=0, maxY=0, minY=0;
    
    //find the max and min values for x, y for all the corners of both images
    for(it1= image1Corners.begin(), it2=image2Corners.begin(); it1<image1Corners.end(); it1++, it2++)
    {
        if(it1->x > maxX)
            maxX = it1->x;
        if(it2->x > maxX)
            maxX = it2->x;
        if(it1->y > maxY)
            maxY = it1->y;
        if(it2->y > maxY)
            maxY = it2->y;
        if(it1->x < minX)
            minX = it1->x;
        if(it2->x < minX)
            minX = it2->x;
        if(it1->y < minY)
            minY = it1->y;
        if(it2->y < minY)
            minY = it2->y;
    }
    map<String, Point2f> newCorners;
    newCorners["Min"] = Point2f(minX,minY);
    newCorners["Max"] = Point2f(maxX,maxY);
    return newCorners;
}
/**
 @function getStitchedImage
 @param image1 & 2 : images we wish to stitch together, HomInv : the homography from image 2 to 1
 @return a Stitched info struct containing a black image with the dimensions of the stitched image, + norm...
 */
static StitchedInfo getStitchedImage(Mat image1, Mat image2, Mat HomInv)
{
    map<String, Point2f> dimensions = getCornersVec(image1, image2, HomInv);
    Mat stitchedImage(abs(dimensions["Max"].y-dimensions["Min"].y),
                      abs(dimensions["Max"].x-dimensions["Min"].x), image1.type(), Scalar::all(0));
    StitchedInfo st;
    st.StitchedImage =stitchedImage;
    st.norm = Point2f(dimensions["Min"].x, dimensions["Min"].y);
    return st;
}
/**
 @function addTo
 @param SI: stitched info struct containing the black stitiched image, Mat image1 : an image in the same plane as the stitched image
 @return bool true if a or b value is in array, false otherwise
 */
static void addTo(StitchedInfo& SI, Mat image1)
{
    for(int r = 0; r< image1.rows; r++)
    {
        for(int c = 0; c< image1.cols; c++)
        {
            SI.StitchedImage.at<Vec3b>(r-SI.norm.y,c-SI.norm.x) = image1.at<Vec3b>(r,c);
        }
    }
}
/**
 @function getInterpolatedValue
 @param Vec3b, v1 and v2
 @return a Vec3b containing the average values of v1 and v2
 */
static Vec3b getInterpolatedValue(Vec3b v1, Vec3b v2)
{
    int g=0, r=0, b=0;
    g = (v1[0]+v2[0])/2;
    r = (v1[1]+v2[1])/2;
    b = (v1[2]+v2[2])/2;
    return Vec3b(g,r,b);
}
/**
 @function projectInto
 @param SI: stitched info struct containing the incomplete stitched image ,
 @param Mat image2 : an image we wish to add into the stitched image, Hom: the homography from stitched image to image 2
 
 */
static void projectInto(StitchedInfo& SI, Mat image2, Mat Hom)
{
    for(int r = SI.norm.y; r < SI.StitchedImage.rows+SI.norm.y; r++)
    {
        for(int c = SI.norm.x; c < SI.StitchedImage.cols+SI.norm.x; c++)
        {
            Point2f temp = project(c,r, Hom);
            if(0<=temp.y && temp.y<image2.rows && 0<=temp.x && temp.x<image2.cols)
            {
                Mat output;
                if(SI.StitchedImage.at<Vec3b>(r-SI.norm.y,c-SI.norm.x) == Vec3b(0,0,0))
                {
                    getRectSubPix(image2, Size_<float>(1, 1), temp, output);
                    SI.StitchedImage.at<Vec3b>(r-SI.norm.y,c-SI.norm.x)= output.at<Vec3b>(0,0);
                }
                else
                {
                    getRectSubPix(image2, Size_<float>(1, 1), temp, output);
                    SI.StitchedImage.at<Vec3b>(r-SI.norm.y,c-SI.norm.x)= getInterpolatedValue(output.at<Vec3b>(0,0),SI.StitchedImage.at<Vec3b>(r-SI.norm.y,c-SI.norm.x));
                }
            }
        }
    }
}
/**
 @function Stitch
 @param img_1&2: images we wish to stitch together
 @param Homographies: map containing Hom and HomInv --> homographies between img_1 and 2
 @return the Mat result of stitching img_1 and 2 given homographies
 */
static Mat Stitch(Mat img_1, Mat img_2, map<String, Mat> Homographies)
{
    StitchedInfo Stitched = getStitchedImage(img_1, img_2, Homographies["HomInv"]);
    addTo(Stitched, img_1);
    projectInto(Stitched, img_2, Homographies["Hom"]);
    return Stitched.StitchedImage;
}
/**
 @function combine
 @param extremas : vector of Max and Min points
 @return map of the max and min points of all the points in the vector
 */
static map<String, Point2f> combine(vector<Point2f> extremas)
{
    vector<Point2f>::iterator it;
    int maxX= 0 , maxY=0, minX=0, minY=0;
    
    for(it= extremas.begin(); it < extremas.end(); it++)
    {
        if(it->x > maxX)
            maxX = it->x;
        if(it->y > maxY)
            maxY = it->y;
        if(it->x < minX)
            minX = it->x;
        if(it->y < minY)
            minY = it->y;
    }
    map<String, Point2f> extrs;
    extrs["Min"] = Point2f(minX,minY);
    extrs["Max"] = Point2f(maxX,maxY);
    
    return extrs;
}
/**
 @function stitchMultiple
 @param image1 : the image of the plan we wish to project into
 @param ims : vector of toStitch structs
 @return Stitched info struct containing the stitched image. 
 */
static StitchedInfo stitchMultiple(Mat image1 , vector<toStitch> images)
{
    vector<Point2f> extremas;
    map<String, Point2f> temp;
    vector<toStitch>::iterator it;
    for(it= images.begin(); it< images.end(); it++)
    {
        temp = getCornersVec(image1, it->image, it->HomToCenter);
        extremas.push_back(temp["Max"]);
        extremas.push_back(temp["Min"]);
    }
    
    map<String, Point2f> dimensions = combine(extremas);
    Mat stitchedImage(abs(dimensions["Max"].y-dimensions["Min"].y),
                      abs(dimensions["Max"].x-dimensions["Min"].x), image1.type(),  Scalar::all(0));
    StitchedInfo st;
    st.StitchedImage =stitchedImage;
    st.norm = Point2f(dimensions["Min"].x, dimensions["Min"].y);
    return st;
}
/**
 @function Feather Helper
 @param val : the uchar of a Vec3b
 @return the value modified by our desired feathering
 */
static bool isInElipse(double c, double r, double x, double y, double percent)
{
    double xTrans = pow((x-(c/2)),2)/pow(((c/2)-(c*percent)),2);
    double yTrans = pow((y-(r/2)),2)/pow(((r/2)-(r*percent)),2);
    if(xTrans+yTrans < 1)
        return true;
    else
        return false;
}
/**
 @function Feather Helper
 @param val : the uchar of a Vec3b
 @return the value modified by our desired feathering
 */
inline static uchar getFVal(uchar val, double c, double r, double x, double y, double percent)
{
    double distX = abs(c-x), distY = abs(r-y);
    if(distX > c/2)
        distX = abs(distX- c);
    if(distY > r/2)
        distY = abs(distY-r);
    
    double transform = 1;
    
    if(distY > distX)
    {
         transform = (r*percent)/(distY);
    }
    else
    {
         transform = (c*percent)/(distX);
    }
    
    return val/transform;
}
/**
 @function Feather Helper
 @param image1 : the image we wish to feather
 @param amount : the degree to which we wish to feather the image
 @return a Feathered version of image 1
 */
static Vec3b getFeatheredVec( double c, double r, Vec3b v, double x, double y)
{
    double p = .23;
    if(!isInElipse(c,r,x,y,.1) )
    {
        return Vec3b( getFVal(v[0],c,r,x,y,p), getFVal(v[1],c,r,x,y,p), getFVal(v[2],c,r,x,y,p));
    }
    else
    {
        return v;
    }
}


/**
 @function getFeatheredImage
 @param image1 : the image we wish to feather
 @param amount : the degree to which we wish to feather the image
 @return a Feathered version of image 1
 */
static Mat getFeatheredImage(Mat image, double feather)
{
    int rows = image.rows, cols = image.cols;
    Mat toReturn(image);
    
    for(int c = 0; c< cols; c++)
    {
        for(int r = 0; r< rows; r++)
        {
             image.at<Vec3b>(r,c) = getFeatheredVec( cols, rows, image.at<Vec3b>(r,c) ,  c,  r);
        }
    }
    return image;
    
}
