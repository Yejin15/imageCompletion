//
//  main.cpp
//  teest
//
//  Created by KimYejin on 18/01/2019.
//  Copyright © 2019 KimYejin. All rights reserved.
//
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <time.h>
#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
struct level{Mat mat; float con_sd;float con_mean;};
struct composited{Mat Lk; Mat alpha;};
Mat FastApprox(int L, Mat pyramid, Mat C_bar, Mat alpha);
Point searchR(Mat current_mat, Mat confidence_map, Point target, int size);
Mat adaptiveN(Mat);
Mat confidence(Mat inverse_alpha);
level levelset(Mat confidenceMap);
composited composition( Mat current, Mat inverse_alpha , Point target, Point source, int size);
int main( int argc, char** argv )
{
    /*
     String imageName( "/Users/maeg/projects/openCV/image/elephant_c.jpg" ); // 이미지 위치
     String f_imageName( "/Users/maeg/projects/openCV/image/elephant_f.jpg" );
     */
    
    String imageName( "/Users/maeg/projects/openCV/new_img/image/boat_small.jpg" ); // 이미지 위치
    String f_imageName( "/Users/maeg/projects/openCV/new_img/image/boat_small_inv_matte.jpg" );
    
    /*
     String imageName( "/image/birds.jpg" ); // 이미지 위치
     String f_imageName( "/image/birds_inv_matte.jpg" );
     */
    
    if( argc > 1)
    {
        imageName = argv[1];
    }
    Mat imageC;
    Mat f_image;
    imageC = imread( imageName, IMREAD_COLOR ); // 파일을 읽어옴
    f_image = imread( f_imageName, IMREAD_COLOR );
    if( imageC.empty() )                      // Check for invalid input -> image에 대해서
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    imageC.convertTo(imageC, CV_32FC3, 1.0/255.0);
    
    // 여기부터 픽셀값 다루는 구간
    int height = imageC.rows;
    int width = imageC.cols;
    cout << height << " "<< width << endl;
    //이미지의 크기를 저장
    
    double t3 = (double)getTickCount(); //시간을 재본다
    Mat C_bar(height, width, CV_32FC3); //  C * inverse_alpha
    Mat inverse_alpha(height, width, CV_8U); // 역알파값
    Mat alpha(height, width, CV_8U); // 원래 알파값
    Mat confidence_map(height, width, CV_32F); //confidence map
    Mat N_size(height, width, CV_8U);//픽셀 별 neighborhood 크기(일단 15로 통일)
    Mat level_set(height, width, CV_32F); // level set
    Mat R_source;//소스 프레그먼트들의 후보
    Mat N = adaptiveN(imageC);
    int size ; //타겟의 neighbor 크기를 가져온다
    float confidence_sd;
    Mat search_test;
    Mat current_mat(height, width, CV_32FC3); // 피라미드결과값이 담길 행렬
    Mat inverse_alpha_out;
    Mat confidence_map_out;
    Mat C_bar_out;
    Mat FastA_out;
    Mat level_set_out ;
    Mat N_out;
    Mat alpha_out;
    Mat channels[3];
    Range range(0,height*width);
    split(f_image, channels);
    alpha = (channels[0] != 255) & (channels[1] != 255) & (channels[2] != 255);
    alpha.convertTo(alpha, CV_32F, 1.0/255.0);
    //alpha 생성
    split(f_image, channels);
    inverse_alpha = (channels[0] == 255) & (channels[1] == 255) & (channels[2] == 255);
    inverse_alpha.convertTo(inverse_alpha, CV_32F, 1.0/255.0);
    //inverse alpha 생성
    
    
    Mat alphaThreeChannels[3]={alpha, alpha, alpha}, alphaThree;
    merge(alphaThreeChannels,3,alphaThree);
    Mat inverse_alphaThreeChannels[3]={inverse_alpha, inverse_alpha, inverse_alpha}, inverse_alphaThree;
    merge(inverse_alphaThreeChannels,3,inverse_alphaThree);
    
    imageC.copyTo(C_bar);//씨바는 원래 이미지
    
    multiply(C_bar, inverse_alphaThree, C_bar);
    //원래 C_bar에 inverse 알파값 곱해서 C_bar 만들기 -> 다시 모르는 부분은 지워준다
    current_mat = C_bar + alphaThree;
    //피라미드 초기값 만들기 : 모르는 부분을 하얗게(1값으로)바꿔준다
    C_bar.convertTo(C_bar_out, CV_8UC3, 255.0);
    imwrite("/Users/maeg/projects/openCV/image/vis/First_Cbar.jpg", C_bar_out);
    float confidence_mean = 0.f;
    int iteration=0;
    
    for (int i=0; i<200; i++){ //*********************************전체 반복문
        iteration++;
        //피라미드 계산 시작
        Mat fast_app;
        fast_app = FastApprox(3, current_mat, C_bar, alpha);
        fast_app.copyTo(current_mat);//approximate결과를 저장
        confidence_map = confidence(inverse_alpha);
        level l = levelset(confidence_map);
        level_set = l.mat;
        confidence_sd = l.con_sd;
        confidence_mean = l.con_mean;
        
        //Search 시작
        double minVal;
        double maxVal;
        Point minLoc;
        Point target;
        minMaxLoc( level_set, &minVal, &maxVal, &minLoc, &target );
        
        N = adaptiveN(current_mat);
        double minN;
        double maxN;
        Point minNLoc;
        Point maxNloc;
        minMaxLoc(N, &minN, &maxN, &minNLoc, &maxNloc);
        size = (int)N.at<float>(target)*3 + 8;
        cout << "size = " << size << endl;
        if (target.x + size > width-1) target.x = width - size -1;
        if (target.x - size < 0) target.x = size;
        if (target.y + size > height-1) target.y = height - size -1;
        if (target.y - size < 0) target.y = size;// 타겟 프레그먼트가 이미지 영역을 벗어나지 않도록
        current_mat.convertTo(search_test, CV_8UC3, 255.0);
        Point source = searchR(current_mat, confidence_map, target, size);
        if (source.x + size > width-1) source.x = width - size -1;
        if (source.x - size < 0) source.x = size;
        if (source.y + size > height-1) source.y = height - size -1;
        if (source.y - size < 0) source.y = size;
        
        
        circle(search_test, target, size, Scalar(255, 255, 0),2);
        circle(search_test, source, size, Scalar(0, 0, 255),1); //search의 결과 확인
        
        
        //compositing 시작
        composited com = composition(current_mat, inverse_alpha, target, source, size);
        Mat lk_temp = com.Lk;//새로운 컬러 조각(가우시안 알파 합 반영)
        Mat new_alpha_temp = com.alpha; //새로운 인벌스 알파 조각
        Mat lk;
        Mat new_alpha;
        lk_temp.copyTo(lk);
        new_alpha_temp.copyTo(new_alpha);
        
        lk.copyTo(C_bar(Rect(target.x-size, target.y-size,size*2+1,size*2+1)));//방금
        new_alpha.copyTo(inverse_alpha(Rect(target.x-size, target.y-size,size*2+1,size*2+1)));
        Mat(1-new_alpha).copyTo(alpha(Rect(target.x-size, target.y-size,size*2+1,size*2+1)));
        inverse_alpha.convertTo(inverse_alpha_out, CV_8U, 255.0);
        imwrite("/Users/maeg/projects/openCV/image/vis/inverse alpha_out.jpg", inverse_alpha_out);
        
        cout<< "iteration =   " << iteration+1 << endl;
        cout<<"confidence mean = " << confidence_mean << endl;
        imshow("output",search_test);
        waitKey(1);
    }//*************************전체 반복문 끝
    
    t3 = ((double)getTickCount() - t3) / getTickFrequency();
    cout << "time data =  " << t3 << " sec" << endl;
    
    alpha.convertTo(alpha_out, CV_8U, 255.0);
    inverse_alpha.convertTo(inverse_alpha_out, CV_8U, 255.0);
    confidence_map.convertTo(confidence_map_out, CV_8U, 255.0);
    C_bar.convertTo(C_bar_out, CV_8UC3, 255.0);
    current_mat.convertTo(FastA_out, CV_8UC3, 255.0);
    level_set_out = level_set / (1.f + confidence_sd);
    level_set_out.convertTo(level_set_out, CV_8U, 255.0);
    N.convertTo(N_out, CV_8U, 255.0);
    
    imwrite("/Users/maeg/projects/openCV/image/alpha.jpg", alpha_out);
    imwrite("/Users/maeg/projects/openCV/image/inverse_alpha.jpg", inverse_alpha_out);
    imwrite("/Users/maeg/projects/openCV/image/level_set.jpg", level_set_out);
    imwrite("/Users/maeg/projects/openCV/image/confidence_map.jpg", confidence_map_out);
    imwrite("/Users/maeg/projects/openCV/image/C_bar.jpg", C_bar_out);
    imwrite("/Users/maeg/projects/openCV/image/pyramidMat.jpg", FastA_out);
    imwrite("/Users/maeg/projects/openCV/image/Ntest.jpg", N_out);
    imwrite("/Users/maeg/projects/openCV/image/Search_test.jpg", search_test);
    waitKey(0); // Wait for a keystroke in the window
    
    return 0;
}
Mat FastApprox(int L, Mat current_mat, Mat C_bar, Mat alpha){
    for (int l=L; l>0; l--){
        float difference = 10000; //수렴하는지 비교할 숫자
        Mat temp(current_mat.rows, current_mat.cols, CV_32FC3);
        while (difference > 3.f){//수렴 판별 -> 실행전후 결과가 거의 같을때까지 반복
            current_mat.copyTo(temp);
            if (l > 1){ //level이 1보다 높을때는 level만큼 피라미드 down/up
                for (int k=0; k<l; k++) pyrDown( current_mat, current_mat, Size( current_mat.cols/2, current_mat.rows/2 ) );
                for (int k=0; k<l; k++) pyrUp( current_mat, current_mat, Size( current_mat.cols*2, current_mat.rows*2 ) );
            }
            else { //마지막 level1에서 가우시안 블러처리
                GaussianBlur(current_mat, current_mat, Size( 9, 9 ), 1,0);
            }
            Range range(0,current_mat.rows*current_mat.cols);
            Mat alpha_3C[3] = {alpha, alpha, alpha}, alpha_3;
            merge(alpha_3C,3,alpha_3);
            
            multiply(current_mat, alpha_3, current_mat);
            current_mat = current_mat + C_bar;
            
            difference = 0;
            parallel_for_(range, [&](const Range& range){
                for(auto i=range.start; i<range.end; i++){
                    difference += pow((current_mat.at<Vec3f>(i)[0] - temp.at<Vec3f>(i)[0]),2);
                    difference += pow((current_mat.at<Vec3f>(i)[1] - temp.at<Vec3f>(i)[1]),2);
                    difference += pow((current_mat.at<Vec3f>(i)[2] - temp.at<Vec3f>(i)[2]),2);
                }
            });
            //cout << "difference : " << difference << endl;
            //cout << "level :" <<  l << endl;
        }
        //cout << "Pass" << endl;
    }
    return current_mat;
}
Mat confidence(Mat inverse_alpha){
    Mat confidence_map(inverse_alpha.rows, inverse_alpha.cols, CV_32F); //confidence map
    Range range(0,confidence_map.rows*confidence_map.cols);
    Mat alpha_squared;
    multiply(inverse_alpha, inverse_alpha, alpha_squared);
    GaussianBlur(alpha_squared, confidence_map, Size(9,9), 0);
    parallel_for_(range, [&](const Range& range){
        for(auto i=range.start; i<range.end; i++){
            if (inverse_alpha.at<float>(i) == 1.f) confidence_map.at<float>(i) = 1.f;
            if (confidence_map.at<float>(i)>1.f) confidence_map.at<float>(i) = 1.f;
        }
    });
    Mat confidence_vis;
    confidence_map.convertTo(confidence_vis, CV_8U, 255.0);
    return confidence_map;
}
level levelset(Mat confidence_map){
    level l;
    
    int height = confidence_map.rows;
    int width = confidence_map.cols;
    Mat level_set(height,width,CV_32F);
    Range range(0,confidence_map.rows*confidence_map.cols);
    float confidence_sum = 0.f;
    for (int y=0; y<confidence_map.rows; y++){
        for (int x=0; x<confidence_map.cols; x++){
            confidence_sum += confidence_map.at<float>(y,x);
        }
    }
    float confidence_mean = float(confidence_sum / (height*width));
    float confidence_var = 0;
    for (int y=0; y<confidence_map.rows; y++){
        for (int x=0; x<confidence_map.cols; x++){
            confidence_var += pow( confidence_map.at<float>(y,x) - confidence_mean, 2);
        }
    }
    //confidence sd구하기
    confidence_var = confidence_var/(width*height);
    float confidence_sd = sqrt(confidence_var);
    parallel_for_(range, [&](const Range& range){
        for(auto i=range.start; i<range.end; i++){
            if (confidence_map.at<float> (i) > confidence_mean) level_set.at<float> (i) = 0;
            else {
                level_set.at<float> (i) = confidence_map.at<float>(i) + (rand()/(float)RAND_MAX)*confidence_sd ;
            }
        }
    });
    l.con_mean = confidence_mean;
    l.con_sd = confidence_sd;
    l.mat = level_set;
    cout << "confidence mean = " << confidence_mean << endl;
    return l;
}
Point searchR(Mat current_mat, Mat confidence_map, Point target, int size){
    //가장 작은 결과값의 source fragment의 위치정보와 크기와 방향정보 반환
    int psize = size*2+1;
    int height = current_mat.rows;
    int width = current_mat.cols;
    //Mat distance(height,width,CV_32F);
    Mat lab(height,width,CV_32FC3);
    cvtColor(current_mat, lab, COLOR_BGR2Lab);//lab값으로 변환
    Mat gradient;
    Laplacian(current_mat, gradient, CV_32FC3);
    Range range(0,current_mat.rows*current_mat.cols);
    
    Range size_range(0,psize*psize);
    Mat l_distance(height,width,CV_32F); // 50개 거르기
    Mat argmin(height,width,CV_32F);//5개 거르기
    Mat target_lab = lab(Rect(target.x-size, target.y-size,psize,psize));
    float target_mean, target_sqsum=0.f, target_var, target_sd;
    target_mean = sum( target_lab )[0]/psize/psize;
    for(auto i=size_range.start; i<size_range.end; i++){
        target_sqsum += pow((target_mean-target_lab.at<Vec3f>(i)[0]),2);
    }
    target_var = target_sqsum/(psize*psize);
    target_sd  = sqrt(target_var);
    l_distance = 1000000000.f;
    for (int y = size; y < height-size; y++) { //l값의 차이 구하는 루프 : 50개 거르기
        for (int x = size; x < width-size; x++) {
            Mat source_lab_temp = lab(Rect(x-size, y-size, psize,psize));
            Mat source_lab;
            source_lab_temp.copyTo(source_lab);
            
            float source_mean, source_sqsum=0.f, source_var, source_sd;
            //타겟과 소스의 밝기 평균 구하기
            source_mean = sum( source_lab )[0]/psize/psize;
            for(auto i=size_range.start; i<size_range.end; i++){
                source_sqsum += pow((source_mean-source_lab.at<Vec3f>(i)[0]),2);
            }
            source_var = source_sqsum/(psize*psize);
            source_sd  = sqrt(source_var);
            
            float h_sqr = 1-sqrt(2*target_sd*source_sd/(target_var+source_var))
            * exp(-(0.25)*pow(target_mean-source_mean,2)/(target_var+source_var));
            l_distance.at<float>(y,x) += h_sqr;
            if (confidence_map.at<float>(y,x)<0.99) l_distance.at<float>(y,x) = 100000000.f;
        }
    }
    
    Mat sort_mat;
    Mat distance_1 = l_distance.reshape(1, l_distance.rows*l_distance.cols);
    sortIdx(distance_1, sort_mat, SORT_EVERY_COLUMN | SORT_ASCENDING);
    Mat target_f = current_mat(Rect(target.x-size, target.y-size,psize,psize));
    Mat image_dx; Sobel( current_mat, image_dx, CV_32F, 1, 0);
    Mat image_dy; Sobel( current_mat, image_dy, CV_32F, 0, 1);
    Mat target_sobelX = image_dx (Rect(target.x-size, target.y-size, psize, psize));
    Mat target_sobelY = image_dy (Rect(target.x-size, target.y-size, psize, psize));
    Mat target_c = confidence_map(Rect(target.x-size, target.y-size,size*2+1,size*2+1));
    
    argmin= 10000.f;
    //    for( int i=0; i<50; i++){ //걸러진 50개에 대해서 distance 구하기
    for (int y = size; y < height-size; y++) { //l값의 차이 구하는 루프 : 50개 거르기
        for (int x = size; x < width-size; x++) {
            //        int index = sort_mat.at<int>(i);
            //        int x = index % l_distance.cols;
            //        int y = int(index / l_distance.cols);
            
            Mat source_f = current_mat(Rect(x-size, y-size, psize, psize)).clone();
            Mat source_c = confidence_map(Rect(x-size, y-size, psize, psize)).clone();
            if( sum( source_c )[0] < 0.99*psize*psize ) continue;
            
            Mat source_sobelX = image_dx (Rect(x-size, y-size, psize, psize));
            Mat source_sobelY = image_dy (Rect(x-size, y-size, psize, psize));
            /*
             
             
             vector<Mat> bgr_target(3), bgr_source(3);
             Mat target_sobelX_b, target_sobelX_g, target_sobelX_r;
             Mat target_sobelY_b, target_sobelY_g, target_sobelY_r;
             Mat source_sobelX_b, source_sobelX_g, source_sobelX_r;
             Mat source_sobelY_b, source_sobelY_g, source_sobelY_r;
             split(target_f, bgr_target);
             split(source_f, bgr_source);
             cv::Sobel(bgr_target[0], target_sobelX_b, CV_32F, 1, 0);
             cv::Sobel(bgr_target[1], target_sobelX_g, CV_32F, 1, 0);
             cv::Sobel(bgr_target[2], target_sobelX_r, CV_32F, 1, 0);
             cv::Sobel(bgr_target[0], target_sobelY_b, CV_32F, 0, 1);
             cv::Sobel(bgr_target[1], target_sobelY_g, CV_32F, 0, 1);
             cv::Sobel(bgr_target[2], target_sobelY_r, CV_32F, 0, 1);
             
             cv::Sobel(bgr_source[0], source_sobelX_b, CV_32F, 1, 0);
             cv::Sobel(bgr_source[1], source_sobelX_g, CV_32F, 1, 0);
             cv::Sobel(bgr_source[2], source_sobelX_r, CV_32F, 1, 0);
             cv::Sobel(bgr_source[0], source_sobelY_b, CV_32F, 0, 1);
             cv::Sobel(bgr_source[1], source_sobelY_g, CV_32F, 0, 1);
             cv::Sobel(bgr_source[2], source_sobelY_r, CV_32F, 0, 1);
             
             */
            
            argmin.at<float>(y,x) = 0.f;
            float weight=0.00000001;
            parallel_for_(Range(0,psize), [&](const Range size_range){
                for(auto yy=size_range.start; yy<size_range.end; yy++) for( int xx=0;xx<psize;xx++){
                    float distance = 0.f;
                    float pixel_value = 0.f;
                    
                    distance += fabs(target_f.at<Vec3f>(yy,xx)[0] - source_f.at<Vec3f>(yy,xx)[0]);
                    distance += fabs(target_f.at<Vec3f>(yy,xx)[1] - source_f.at<Vec3f>(yy,xx)[1]);
                    distance += fabs(target_f.at<Vec3f>(yy,xx)[2] - source_f.at<Vec3f>(yy,xx)[2]);
                    //gradient distance 구하기
                    distance += fabs(target_sobelX.at<Vec3f>(yy,xx)[0] - source_sobelX.at<Vec3f>(yy,xx)[0]);
                    distance += fabs(target_sobelX.at<Vec3f>(yy,xx)[1] - source_sobelX.at<Vec3f>(yy,xx)[1]);
                    distance += fabs(target_sobelX.at<Vec3f>(yy,xx)[2] - source_sobelX.at<Vec3f>(yy,xx)[2]);
                    distance += fabs(target_sobelY.at<Vec3f>(yy,xx)[0] - source_sobelY.at<Vec3f>(yy,xx)[0]);
                    distance += fabs(target_sobelY.at<Vec3f>(yy,xx)[1] - source_sobelY.at<Vec3f>(yy,xx)[1]);
                    distance += fabs(target_sobelY.at<Vec3f>(yy,xx)[2] - source_sobelY.at<Vec3f>(yy,xx)[2]);
                    pixel_value = distance * source_c.at<float>(yy,xx) * target_c.at<float>(yy,xx)
                    + (target_c.at<float>(yy,xx)-source_c.at<float>(yy,xx)) * target_c.at<float>(yy,xx);
                    weight+=source_c.at<float>(yy,xx);
                    argmin.at<float>(y,x) += pixel_value;
                }
            });
            //        argmin.at<float>(y,x) /=weight;
        }
        
    }
    double min_s;
    double max_s;
    Point minLoc_s;
    Point maxLoc_s;
    minMaxLoc( argmin, &min_s, &max_s, &minLoc_s, &maxLoc_s );
    return minLoc_s;
    /*
     Mat sort_mat2, sort_test2;
     Mat argmin_1 = argmin.reshape(1, argmin.rows*argmin.cols);
     
     sortIdx(argmin_1, sort_mat2, SORT_EVERY_COLUMN | SORT_ASCENDING);
     cv::sort(argmin_1, sort_test2, SORT_EVERY_COLUMN | SORT_ASCENDING);
     for (int i=0; i<5; i++){ //5개 중 lab 가장 가까운 것 찾기
     
     int index = sort_mat2.at<int>(i);
     int x = index % argmin.cols;
     int y = int(index / argmin.cols);
     
     Mat target_lab = lab(Rect(target.x-size, target.y-size,psize,psize));
     Mat source_lab_temp = lab(Rect(x-size, y-size, psize,psize));
     Mat source_lab;
     source_lab_temp.copyTo(source_lab);
     
     float target_sum=0.f, target_mean, target_sqsum=0.f, target_var, target_sd;
     float source_sum=0.f, source_mean, source_sqsum=0.f, source_var, source_sd;
     //타겟과 소스의 밝기 평균 구하기
     parallel_for_(size_range, [&](const Range size_range){
     for(auto i=size_range.start; i<size_range.end; i++){
     target_sum += target_lab.at<Vec3f>(i)[0];
     source_sum += source_lab.at<Vec3f>(i)[0];
     }
     });
     target_mean = target_sum/(psize*psize);
     source_mean = source_sum/(psize*psize);
     parallel_for_(size_range, [&](const Range size_range){//분산 구하기
     for(auto i=size_range.start; i<size_range.end; i++){
     target_sqsum += pow((target_mean-target_lab.at<Vec3f>(i)[0]),2);
     source_sqsum += pow((source_mean-source_lab.at<Vec3f>(i)[0]),2);
     }
     });
     target_var = target_sqsum/(psize*psize);
     source_var = source_sqsum/(psize*psize);
     target_sd = sqrt(target_var);
     source_sd = sqrt(source_var);
     
     float h_sqr = 0.f;
     l_distance.at<float>(y,x) = 0.f;
     
     parallel_for_(size_range, [&](const Range size_range){//분산 구하기
     for(auto i=size_range.start; i<size_range.end; i++){
     h_sqr = 1-sqrt(2*target_sd*source_sd/(target_var+source_var))
     * exp(-(0.25)*pow(target_mean-source_mean,2)/(target_var+source_var));
     l_distance.at<float>(y,x) += h_sqr;
     }
     });
     }
     double min_s;
     double max_s;
     Point minLoc_s;
     Point maxLoc_s;
     minMaxLoc( l_distance, &min_s, &max_s, &minLoc_s, &maxLoc_s );
     return minLoc_s;*/
}
Mat adaptiveN( Mat mat){//pyramid mat을 받아와서
    Range range(0,mat.rows*mat.cols);
    Mat output(mat.rows, mat.cols, CV_32F);
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {//pyramid mat의 모든 픽셀에 대해서
            Mat extreme(3, 3, CV_32FC3);
            if (0 <= y-1 && y+1 < mat.rows && 0 <= x-1 && x+1 < mat.cols) extreme = mat(Rect(x-1,y-1,3,3));
            else extreme = mat(Rect(x,y,1,1));
            extreme.reshape(1);
            double minVal;
            double maxVal;
            minMaxIdx(extreme, &minVal, &maxVal);
            output.at<float> (y,x) = 1/(maxVal-minVal+0.01);
        }
    }
    double minVal;
    double maxVal;
    minMaxIdx(output, &minVal, &maxVal);
    parallel_for_(range, [&](const Range& range){
        for(auto i=range.start; i<range.end; i++){
            output.at<float> (i) = (output.at<float> (i)-minVal)/(maxVal-minVal);
        }
    });
    //cout << minVal<<" & "<<maxVal<<endl;
    return output;
}
composited composition(Mat img, Mat alpha_img, Point target, Point source, int radius){
    //진행중인 이미지와 타겟 이미지와 소스 이미지를 각각 가져와서 바뀐 전체 이미지를 리턴
    
    int size = radius*2+1;
    Mat alpha_g2, alpha_g3;
    Mat img_l1, img_l2, img_l3;
    Mat target_out;
    Mat source_out;
    
    pyrDown(alpha_img, alpha_g2, Size(alpha_img.cols/2, alpha_img.rows/2));
    pyrDown(alpha_g2, alpha_g3, Size(alpha_img.cols/4, alpha_img.rows/4));
    pyrUp(alpha_g2, alpha_g2, Size(alpha_img.cols,alpha_img.rows));
    pyrUp(alpha_g3, alpha_g3, Size(alpha_img.cols/2,alpha_img.rows/2));
    pyrUp(alpha_g3, alpha_g3, Size(alpha_img.cols,alpha_img.rows));
    
    Mat TA1, TA2, TA3, alpha_f1, alpha_f2, alpha_f3;
    Mat SA1, SA2, SA3, alpha_b1, alpha_b2, alpha_b3;
    Mat alpha_m1, alpha_m2, alpha_m3;
    
    TA1 =alpha_img(Rect(target.x-radius, target.y-radius,size,size));
    TA2 =alpha_g2(Rect(target.x-radius, target.y-radius,size,size));
    TA3 =alpha_g3(Rect(target.x-radius, target.y-radius,size,size));
    SA1 =alpha_img(Rect(source.x-radius, source.y-radius,size,size));
    SA2 =alpha_g2(Rect(source.x-radius, source.y-radius,size,size));
    SA3 =alpha_g3(Rect(source.x-radius, source.y-radius,size,size));
    
    TA1.copyTo(alpha_f1);
    TA2.copyTo(alpha_f2);
    TA3.copyTo(alpha_f3);
    SA1.copyTo(alpha_b1);
    SA2.copyTo(alpha_b2);
    SA3.copyTo(alpha_b3);
    alpha_m1 = 1.f - alpha_f1;
    alpha_m2 = 1.f - alpha_f2;
    alpha_m3 = 1.f - alpha_f3;
    
    Mat alpha_f1_out;
    Mat alpha_b1_out;
    alpha_f1.convertTo(alpha_f1_out, CV_8U, 255.0);
    alpha_b1.convertTo(alpha_b1_out, CV_8U, 255.0);
    
    target_out = img(Rect(target.x-radius, target.y-radius,size,size));
    source_out = img(Rect(source.x-radius, source.y-radius,size,size));
    
    Mat search_test;
    img.convertTo(search_test, CV_8UC3, 255.0);
    
    circle(search_test, target, size,2);
    circle(search_test, source, size,1); //search의 결과 확인
    
    pyrDown(img, img_l1, Size(img.cols/2, img.rows/2));
    pyrDown(img_l1, img_l2, Size(img.cols/4, img.rows/4));
    pyrUp(img_l1, img_l1, Size(img.cols,img.rows));
    pyrUp(img_l2, img_l2, Size(img.cols/2,img.rows/2));
    pyrUp(img_l2, img_l2, Size(img.cols,img.rows));
    img_l2.copyTo(img_l3);
    img_l2 = img_l2 - img_l1;
    img_l1 = img - img_l1;
    
    Mat l1, l2, l3;
    
    img_l1.convertTo(l1, CV_8UC3, 255.0);
    img_l2.convertTo(l2, CV_8UC3, 255.0);
    img_l3.convertTo(l3, CV_8UC3, 255.0);
    
    Mat TL1, TL2, TL3;
    Mat SL1, SL2, SL3;
    TL1 =img_l1(Rect(target.x-radius, target.y-radius,size,size));
    TL2 =img_l2(Rect(target.x-radius, target.y-radius,size,size));
    TL3 =img_l3(Rect(target.x-radius, target.y-radius,size,size));
    SL1 =img_l1(Rect(source.x-radius, source.y-radius,size,size));
    SL2 =img_l2(Rect(source.x-radius, source.y-radius,size,size));
    SL3 =img_l3(Rect(source.x-radius, source.y-radius,size,size));
    
    Mat lap_f1, lap_f2, lap_f3;
    Mat lap_b1, lap_b2, lap_b3;
    
    TL1.copyTo(lap_f1);
    TL2.copyTo(lap_f2);
    TL3.copyTo(lap_f3);
    SL1.copyTo(lap_b1);
    SL2.copyTo(lap_b2);
    SL3.copyTo(lap_b3);
    
    composited result;
    Mat alpha_out(size,size,CV_32F); // 내보낼 알파
    Mat target_m1(size,size,CV_32F); // 1-타겟의 알파
    Range range(0,size*size);
    
    Mat alpha1(size,size,CV_32F), alpha2(size,size,CV_32F), alpha3(size,size,CV_32F);
    Mat alpha_result(size,size,CV_32F);
    
    parallel_for_(range, [&](const Range& range){
        for(auto i=range.start; i<range.end; i++){
            alpha1.at<float>(i) = alpha_f1.at<float>(i) + alpha_b1.at<float>(i) * alpha_m1.at<float>(i);
            alpha2.at<float>(i) = alpha_f2.at<float>(i) + alpha_b2.at<float>(i) * alpha_m2.at<float>(i);
            alpha3.at<float>(i) = alpha_f3.at<float>(i) + alpha_b3.at<float>(i) * alpha_m3.at<float>(i);
        }
    }); //가우시안 알파 각 단계별 계산
    
    alpha3.copyTo(alpha_result);
    Mat alpha_result_out;
    alpha_result.convertTo(alpha_result_out, CV_8U,255.0);
    
    
    Mat lk1(size,size,CV_32FC3),lk2(size,size,CV_32FC3),lk3(size,size,CV_32FC3);
    Mat lk(size,size,CV_32FC3);
    
    Mat alpha_m1_3C[3] = {alpha_m1, alpha_m1, alpha_m1}, alpha_m1_3;
    merge(alpha_m1_3C,3,alpha_m1_3);
    Mat alpha_m2_3C[3] = {alpha_m2, alpha_m2, alpha_m2}, alpha_m2_3;
    merge(alpha_m2_3C,3,alpha_m2_3);
    Mat alpha_m3_3C[3] = {alpha_m3, alpha_m3, alpha_m3}, alpha_m3_3;
    merge(alpha_m3_3C,3,alpha_m3_3);
    Mat alpha_f1_3C[3] = {alpha_f1, alpha_f1, alpha_f1}, alpha_f1_3;
    merge(alpha_f1_3C,3,alpha_f1_3);
    Mat alpha_f2_3C[3] = {alpha_f2, alpha_f2, alpha_f2}, alpha_f2_3;
    merge(alpha_f2_3C,3,alpha_f2_3);
    Mat alpha_f3_3C[3] = {alpha_f3, alpha_f3, alpha_f3}, alpha_f3_3;
    merge(alpha_f3_3C,3,alpha_f3_3);
    Mat alpha_b1_3C[3] = {alpha_b1, alpha_b1, alpha_b1}, alpha_b1_3;
    merge(alpha_b1_3C,3,alpha_b1_3);
    Mat alpha_b2_3C[3] = {alpha_b2, alpha_b2, alpha_b2}, alpha_b2_3;
    merge(alpha_b2_3C,3,alpha_b2_3);
    Mat alpha_b3_3C[3] = {alpha_b3, alpha_b3, alpha_b3}, alpha_b3_3;
    merge(alpha_b3_3C,3,alpha_b3_3);
    
    Mat lk1_term1, lk1_term2;
    multiply(lap_f1, alpha_f1_3, lk1_term1);
    multiply(lap_b1, alpha_b1_3, lk1_term2);
    multiply(lk1_term2, alpha_m1_3, lk1_term2);
    lk1 = lk1_term1 + lk1_term2;
    
    Mat lk2_term1, lk2_term2;
    multiply(lap_f2, alpha_f2_3, lk2_term1);
    multiply(lap_b2, alpha_b2_3, lk2_term2);
    multiply(lk2_term2, alpha_m1_3, lk2_term2);
    lk2 = lk2_term1 + lk2_term2;
    
    Mat lk3_term1, lk3_term2;
    multiply(lap_f3, alpha_f3_3, lk3_term1);
    multiply(lap_b3, alpha_b3_3, lk3_term2);
    multiply(lk3_term2, alpha_m3_3, lk3_term2);
    lk3 = lk3_term1 + lk3_term2;
    //라플라시안의 각 단계별 계산(Lk(Cout))
    
    lk = lk1 + lk2 + lk3;
    
    Mat alpha1_out, alpha2_out, alpha3_out;
    Mat lk1_out, lk2_out, lk3_out, lk_out;
    
    alpha1.convertTo(alpha1_out, CV_8U, 255.0);
    alpha2.convertTo(alpha2_out, CV_8U, 255.0);
    alpha3.convertTo(alpha3_out, CV_8U, 255.0);
    lk1.convertTo(lk1_out, CV_8UC3, 255.0);
    lk2.convertTo(lk2_out, CV_8UC3, 255.0);
    lk3.convertTo(lk3_out, CV_8UC3, 255.0);
    lk.convertTo(lk_out, CV_8UC3, 255.0);
    target_out.convertTo(target_out, CV_8UC3, 255.0);
    source_out.convertTo(source_out, CV_8UC3, 255.0);
    
    result.Lk = lk;
    result.alpha = alpha_result;
    
    return result; //R(C_bar)반환
}
