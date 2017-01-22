/*
Demo of ica.hpp
Author : François-Xavier Derue
francois.xavier.derue<at>gmail.com
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "ica.hpp"
#include <time.h>


using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    
	// Create a Mixed signal
	/*vector<string> vSignalToMixPathName;
	vSignalToMixPathName.push_back("E:/Musics/Divers/hillaryVoice.wav");
	vSignalToMixPathName.push_back("E:/Musics/Divers/trumpVoice.wav");

	string mixPathName = "mixSignal";
	createMixAudioSignal(vSignalToMixPathName,mixPathName);*/

	// Split a Mixed signal
	vector<string> vMixSignalPathName;
	vMixSignalPathName.push_back("E:/Musics/Divers/mixHillaryTrump0.wav");
	vMixSignalPathName.push_back("E:/Musics/Divers/mixHillaryTrump1.wav");

	string splitSignalPathName = "splitSignal";
	splitMixAudioSignal(vMixSignalPathName, splitSignalPathName);
    return 0;
}
