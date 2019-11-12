//
//  bonus2Main.cpp
//  OpenCV-Template
//
//  Created by Benjamin Therien on 2019-03-23.
//  Copyright © 2019 Benjamin Therien. All rights reserved.
//

#include "CornerDescriptor.cpp"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <vector>


using namespace std;
using namespace cv;

int main()
{
    
    string image1Path = "../../../../project_images/Rainier1.png";
    Mat img_1 = imread(image1Path, CV_LOAD_IMAGE_COLOR);
    
    string image2Path = "../../../../project_images/Rainier2.png";
    Mat img_2 = imread(image2Path, CV_LOAD_IMAGE_COLOR);
    
    
    imshow("image",getFeatheredImage(img_1, 0));
    waitKey(0);
    
    return 0;
}
