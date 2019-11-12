
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <unistd.h>

using namespace cv;
using namespace std;


//euclidean distance between 2 points
static inline double distance(Point2f a,Point2f b)
{
    return sqrt(pow(a.x-b.x,2) + pow(a.y-b.y,2));
}

//euclidean distance between match points in 2 images placed side by side
static inline double offsetDistance(Point2f a,Point2f b, int displacement)
{
    return sqrt(pow(a.x-b.x-displacement,2) + pow(a.y-b.y,2));
}

/**
 * @struct: MFHolder
 * Use --> Container for properties of 2 matches used to improve the performace of ransac
 */
struct MFHolder
{
    Point2f from, to;
    int colDisplacement;
    double length, angle;
    MFHolder(){};
    
    MFHolder(int displacement, Point2f a, Point2f b)
    {
        from =a;
        to = a;
        colDisplacement = displacement;
        length = offsetDistance(a, b, displacement);
        angle = asin(sqrt(pow(length, 2) - pow(abs(a.x - b.x -displacement),2))/length)*180/CV_PI;
    };
};

/**
 @function medianDistanceFilter
 @param arr: array of distances, size of the array, a threshold value
 @return true if each distance is less than the threshold value away from the median distance value
 */
static bool medianDistanceFilter(double *arr, int size, double threshold)
{
    sort(arr,arr+size);
    int medIndex = (size/2) -1;
    for(int i=0; i< size; i++)
    {
        if( abs(arr[i] - arr[medIndex]) > threshold)
            return false;
    }
    return true;
}

/**
 @function medianAngleFilter
 @param arr : array of MFHolder, size : length of the array, a threshold value
 @return true if each angle is less than the threshold value away from the median angle value
 */
static bool medianAngleFilter(MFHolder *arr, int size, double threshold)
{
    double a[size];
    for(int i=0; i<size; i++)
    {
        a[i] = arr[i].angle;
    }
    return medianDistanceFilter(a, size, threshold);
}

/**
 @function project
 @param double x coordinate, double y coordinate, homography matrix
 @return the values of the point multiplied by the homography
 **/
static Point2f project(double x, double y, Mat H)
{
    //setVars
    Mat vec = Mat(3,1, H.type()), result = Mat(3,1, H.type());
    vec.at<double>(0,0) = x;
    vec.at<double>(0,1) = y;
    vec.at<double>(0,2) = 1;
    //Implement Matrix Mul
    for(int r=0; r<3 ; r++)
    {
        float temp=0;
        for(int c=0; c<3 ; c++)
        {
            temp = temp + H.at<double>(r,c)* vec.at<double>(0,c);
        }
        result.at<double>(0,r) = temp;
    }
    
    //Normalize
    double normalize = result.at<double>(0,2);
    result.at<double>(0,0) = result.at<double>(0,0)/normalize;
    result.at<double>(0,1) = result.at<double>(0,1)/normalize;
    return Point2f(result.at<double>(0,0),result.at<double>(0,1));
}
/**
 @function computeInlierCount
 @param matches1 & 2 the corresponding mathes between the 2 images, thresh: the threshold value
 @return icount : number of unique points that lie a threshold distance away from their projected counterparts
 */
static int computeInlierCount(Mat H, vector<Point2f> matches1,vector<Point2f> matches2, double thresh)
{
    int iCount=0;
    if(matches1.size() <= matches2.size())
    {
        vector<Point2f> inliers1, inliers2;
        vector<Point2f>::iterator it1, it2;
        it2= matches2.begin();
        for(it1 = matches1.begin(); it1<matches1.end();it1++,it2++)
        {
            if(distance(project(it1->x,it1->y,H),*it2) < thresh && find(inliers1.begin(), inliers1.end(), *it1) == inliers1.end() && find(inliers2.begin(), inliers2.end(), *it2) == inliers2.end())
            {
                iCount++;
                inliers1.push_back(*it1);
                inliers2.push_back(*it2);
            }
        }
    }
    else
    {
        cout<<"error in computeInlierCount"<<endl;
        return -1;
    }
    return iCount;
}

/**
 @function getInliers
 @use fills inliers1 & 2 with the appropriate points
 */
static void getInliers(Mat H, vector<Point2f> matches1,vector<Point2f> matches2, double threshold, vector<Point2f> &inliers1, vector<Point2f> &inliers2)
{
    if(H.rows==0 || H.cols==0)
    {
        cout<<"Homography not found"<<endl;
    }
    vector<Point2f>::iterator it1, it2;
    int iCount=0;
    if(matches1.size() <= matches2.size())
    {
        it2= matches2.begin();
        for(it1 = matches1.begin(); it1<matches1.end();it1++,it2++)
        {
            if((distance(project(it1->x,it1->y,H),*it2) < threshold) && find(inliers1.begin(), inliers1.end(), *it1) == inliers1.end() && find(inliers2.begin(), inliers2.end(), *it2) == inliers2.end())
            {
                iCount++;
                inliers1.push_back(*it1);
                inliers2.push_back(*it2);
            }
            
        }
    }
}

/**
 @function isInArr
 @param a : array of ints, size : lenght of array, x : compare value
 @return bool true if value is in array, false otherwise
 */
static bool isInArr(int x, int *a, int size)
{
    for(int i=0; i<size; i++)
    {
        if(x == a[i])
            return true;
    }
    return false;
}

/**
 @function isInArr
 @param arr : array of MFHolder, size : length of array, a,b : compare values
 @return bool true if a or b value is in array, false otherwise
 */
static bool isInArr2(Point2f a , Point2f b, MFHolder* arr, int size)
{
    for(int i=0; i<size; i++)
    {
            if(a == arr[i].from || b == arr[i].to)
                return true;
    }
    return false;
}

/**
 @function ransac
 @param matches1&2 : corresponding matching points, double thresholds : various threshs, iteration count: number of iteration to run,
 @return map containing the homography computed from the inliers of the best homography, and its inverse
 */
static map<String, Mat> ransac(vector<Point2f> matches1, vector<Point2f> matches2, double threshold, int iterationCount, int displacement, double MDFThreshold, double angleThreshold )
{
    //init necessary data structures
    int* helper= new int[4];
    unsigned long numMatches = matches1.size();
    Point2f assocMatches[numMatches][2];
    int maxInlierCount=0, maxInlierIndex=-1;
    Mat Homographies[iterationCount];
    bool passed =false;
    
    
    vector<Point2f> randomSapleI1, randomSapleI2;
    vector<Point2f>::iterator it1, it2;
    it1 = matches1.begin();
    it2 = matches2.begin();
    
    //sets up array of associated matches to facilitate randomization
    for(int i=0; i<matches1.size(); i++)
    {
        assocMatches[i][0] = *it1;
        assocMatches[i][1] = *it2;
        it1++;
        it2++;
    }

    
    //ransac iterations: the magic happens here!
    for(int i=0; i<iterationCount; i++)
    {
        randomSapleI1.clear();
        randomSapleI2.clear();
        int count=0;
        passed=false;
        
        //get vects of random matching points:
        while(!passed)
        {
            double distances[4];
            MFHolder toFilter[4];
            
            while(count<4)
            {
                int x = rand() % numMatches;
                if(!isInArr(x, helper, count) && !isInArr2(assocMatches[x][0], assocMatches[x][1], toFilter, count))
                {
                    randomSapleI1.push_back(assocMatches[x][0]);
                    randomSapleI2.push_back(assocMatches[x][1]);
                    
                    helper[count] = x;
                    distances[count]= offsetDistance(assocMatches[x][0],assocMatches[x][1], displacement);
                    toFilter[count] = MFHolder(displacement,assocMatches[x][0],assocMatches[x][1]);
                }
                else
                {
                    count--;
                }
                count++;
            }
            count=0;
            
            //filter by median angle and distance
            if(medianAngleFilter(toFilter, 4, angleThreshold))
                passed = true;
            else
                passed = medianDistanceFilter(distances, 4, MDFThreshold);
        }
        
        Mat Homography = findHomography(randomSapleI1, randomSapleI2, 0);
        int tempCount = computeInlierCount(Homography, matches1, matches2, threshold);
        Homographies[i] = Homography;
        
        if(tempCount > maxInlierCount)
        {
            maxInlierCount = tempCount;
            maxInlierIndex = i;
        }
    }
    
    //compute the best homography on all inliers
    randomSapleI1.clear();
    randomSapleI2.clear();
    
    //fill random sample 1 and 2 with matches that are inliers to the best homography found above
    getInliers(Homographies[maxInlierIndex], matches1, matches2, threshold, randomSapleI1, randomSapleI2);
    
    map<String, Mat> homs;
    homs["Hom"] = findHomography(randomSapleI1, randomSapleI2, 0);
    homs["HomInv"] = findHomography(randomSapleI2, randomSapleI1, 0);
    
    //error messages
    if(homs["Hom"].rows==0 || homs["Hom"].cols==0)
    {
        cout<<"Homography not found"<<endl;
    }
    if(homs["HomInv"].rows==0 || homs["HomInv"].cols==0)
    {
        cout<<"Inverse Homography not found"<<endl;
    }
    
    return homs;
}


