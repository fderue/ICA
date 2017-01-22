/*
Implementation of the cocktail party problem :
Separate N mixed audio signal into N separated audio signal

Project for the course INF8703 at Polymtl
Author : François-Xavier Derue
francois.xavier.derue<at>gmail.com
*/

#pragma once

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void createMixAudioSignal(const vector<string>& vSignalToMixPathName,const string outMixPathName);
void splitMixAudioSignal(const vector<string>& vMixSignalPathName, const string splitSignalPathName);
Mat runICA(const Mat& x, const int maxIt = 1000, const int typeOfOp = 1, const float alpha = 1);
void writeRow2WavFile(const Mat& x, const string pathName);
Mat loadWavFile2Mat(const vector<string>& vWavFilePathName);

static void updateW(const Mat& X, Mat& w, Mat& g, Mat& g_prime, int op, float a);
static void compute_g_gp(const Mat& X, Mat& w, Mat& g, Mat& g_prime, int op, float a);
static void remMean(Mat &x, Mat &means);
static Mat myCov(const Mat& x);
static Mat whitening(const Mat& x);
