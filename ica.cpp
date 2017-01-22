#include "ica.hpp"
#include <sndfile.hh>
#include <time.h>

void createMixAudioSignal(const vector<string>& vSignalToMixPathName, const string outMixPathName){
	int nbSignal = vSignalToMixPathName.size();
	if (nbSignal < 2){
		cerr << "not enough signal to mix, min 2 signal required" << endl;
		exit(EXIT_FAILURE);
	}
	Mat matAudioSignal = loadWavFile2Mat(vSignalToMixPathName);
	// Create mixing matrix with random coefficient
	cv::theRNG().state = time(NULL);
	Mat A(nbSignal, nbSignal, CV_32F, Scalar(0));
	cv::randu(A, Scalar(-1), Scalar(1));
	cout << "Mixing matrix " << A << endl;
	
	// Create Mix signals and write to .wav file
	Mat x = A*matAudioSignal;
	writeRow2WavFile(x, outMixPathName);
}

void splitMixAudioSignal(const vector<string>& vMixSignalPathName, const string splitSignalPathName){
	int nbSignal = vMixSignalPathName.size();
	if (nbSignal < 2){
		cerr << "not enough mixed signal given, min 2 signal required" << endl;
		exit(EXIT_FAILURE);
	}
	Mat mixedAudioSignal = loadWavFile2Mat(vMixSignalPathName);
	Mat y = runICA(mixedAudioSignal);
	writeRow2WavFile(y, splitSignalPathName);

}

Mat loadWavFile2Mat(const vector<string>& vWavFilePathName){
	int nbSignal = vWavFilePathName.size();
	// Load audio signal
	vector<int> vNbFrames;
	vector<int> vNbChannels;
	vector<float*> vpAudioData;
	for (int i = 0; i < nbSignal; i++){
		SF_INFO audioInfo;
		audioInfo.format = 0;
		SNDFILE* audioFile = sf_open(vWavFilePathName[i].c_str(), SFM_READ, &audioInfo);
		if (audioFile == nullptr) cerr << "Impossible to open audio file " << vWavFilePathName[i] << endl;
		else{
			float* pAudioData = new float[audioInfo.frames*audioInfo.channels];
			auto nbFrames = sf_readf_float(audioFile, pAudioData, audioInfo.frames);
			vNbFrames.push_back(nbFrames);
			vNbChannels.push_back(audioInfo.channels);
			vpAudioData.push_back(pAudioData);
			sf_close(audioFile);
		}
	}
	int maxNbFrame = *std::max_element(vNbFrames.begin(), vNbFrames.end());
	Mat matAudioSignal(nbSignal, maxNbFrame, CV_32F, Scalar(0));
	for (int signalIdx = 0; signalIdx < nbSignal; signalIdx++){
		float* pAudioSignalRow = matAudioSignal.ptr<float>(signalIdx);
		for (int frameIdx = 0; frameIdx < vNbFrames[signalIdx]; frameIdx++){
			pAudioSignalRow[frameIdx] = vpAudioData[signalIdx][frameIdx*vNbChannels[signalIdx]];
		}
	}
	return matAudioSignal;
}

void writeRow2WavFile(const Mat& x, const string pathName){
	const int format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
	int fs = 44100;
	for (int i = 0; i < x.rows; i++){
		string outName = pathName + to_string(i) + ".wav";
		SndfileHandle outFile(outName, SFM_WRITE, format, 1, fs);
		if (!outFile) cerr << "impossible to write" << endl;
		else outFile.write(x.ptr<float>(i), x.cols);
	}
}

Mat runICA(const Mat& x, const int maxIt, const int typeOfOp, const float alpha){
	int M = x.rows;
	int D = x.cols;
	Mat y = Mat(x.size(), CV_32F, Scalar(0)); //output : demixing signals
	Mat W = Mat(M, M, CV_32F, Scalar(0)); // estimate of channel A^-1

	//===== whithening ===== 
	Mat V = whitening(x); //whitening
	Mat mixedMean = Mat(M, 1, CV_32F, Scalar(0));
	Mat x_mean = x.clone();
	remMean(x_mean, mixedMean); // remove the mean to center the signal
	Mat X = V*x;

	//=== Fixed Point algorithm===
	Mat B = Mat(M, M, CV_32F, Scalar(0));
	Mat g = Mat(D, 1, CV_32F, Scalar(0));
	Mat g_prime = Mat(D, 1, CV_32F, Scalar(0));
	for (int i = 0; i<M; i++) {
		// initialization of w
		Mat w = Mat(M, 1, CV_32F, Scalar(0));
		float* ptrw = w.ptr<float>();
		for (int l = 0; l<M; l++){
			ptrw[l] = 0.5;//rng.gaussian(1);
		}
		w = w - B*B.t()*w; // décorrelation between output
		divide(w, norm(w), w);

		Mat wOld = Mat::zeros(w.size(), CV_32F);
		Mat wOld2 = Mat::zeros(w.size(), CV_32F);

		//== start of the fixed-point algo ====
		float eps = 0.0001;
		for (int j = 0; j<maxIt; j++){
			w = w - B*B.t()*w;
			divide(w, norm(w), w);
			if (norm(w - wOld)<eps || norm(w + wOld)<eps){
				break; // convergence
			}
			wOld = w.clone();
			updateW(X, w, g, g_prime, typeOfOp, alpha);
		}

		for (int b = 0; b<M; b++){
			B.at<float>(b, i) = w.at<float>(b, 0);
		}
		Mat w_withe = w.t()*V;
		for (int m = 0; m<M; m++){
			W.at<float>(i, m) = w_withe.at<float>(0, m);
		}
	}
	Mat mxMean = Mat::ones(1, D, CV_32F);
	y = W*x + W*mixedMean*mxMean;
	return y;
}

static void updateW(const Mat& X, Mat& w,Mat& g, Mat& g_prime, int op, float a){
    compute_g_gp(X,w,g,g_prime,op,a); 
    Mat X_g = X*g;
    Mat beta = w.t()*X_g;
    float beta_f = beta.at<float>(0,0);
    Mat Num = X_g-beta_f*w;
    float Den = sum(g_prime).val[0]-beta_f;
    float Den_1 = 1.f/Den;
    w = w-((X_g-beta_f*w)*Den_1);
    divide(w, norm(w), w);
}

static void compute_g_gp(const Mat& X, Mat& w,Mat& g, Mat& g_prime, int op, float a)
{
    Mat Xtw = X.t()*w; // ok
    
    switch (op) {
        case 0:
        {
            Mat tanh_Xtw;
            exp(2*a*Xtw, tanh_Xtw);
            add(tanh_Xtw, 1, tanh_Xtw);
            divide(2, tanh_Xtw, tanh_Xtw);
            subtract(1, tanh_Xtw, tanh_Xtw);
            g = tanh_Xtw.clone();
            pow(tanh_Xtw,2,tanh_Xtw);
            subtract(1, tanh_Xtw, g_prime);
            g_prime = a*g_prime;
            break;
        }
        case 1:
        {
            
            Mat Xtw2;
            pow(Xtw, 2, Xtw2);
            Mat ex;
            exp(-a/2.f*Xtw2, ex);
            multiply(ex, Xtw, g);
            subtract(1, a*Xtw2, g_prime);
            multiply(g_prime, ex, g_prime);
            break;
                            
        }
        default:
            break;
    }
}

static void remMean(Mat &x, Mat &means){
    int M = x.rows; //# signals
    int D = x.cols; //# dimension
    for(int i=0; i<M; i++){
        float* ptrX_cols = x.ptr<float>(i);
        for(int j=0; j<D; j++)
        {
            means.at<float>(i,0)+=ptrX_cols[j];
        }
        means.at<float>(i, 0)/=D;
    }
    for(int i=0; i<M; i++){
        float* ptrX_cols = x.ptr<float>(i);
        for(int j=0; j<D; j++)
        {
            ptrX_cols[j]-=means.at<float>(i,0);
        }
    }
}

static Mat myCov(const Mat& x){
    int M = x.rows; //# signals
    int D = x.cols; //# dimension
    //======= covariance matrix of x =========
    Mat Cxx = Mat(M,M,CV_32F,Scalar(0));
    //==compute mean==
    Mat means = Mat(M,1,CV_32F,Scalar(0));
    for(int i=0; i<M; i++)
    {
        const float* ptrX_cols = x.ptr<float>(i);
        for(int j=0; j<D; j++)
        {
            means.at<float>(i,0)+=ptrX_cols[j];
        }
        means.at<float>(i, 0)/=D;
    }
    //==compute coefficient Cxixj==
    for(int i=0; i<M; i++) {
        for(int j=0; j<M; j++){
            float mean_i = means.at<float>(i,0);
            float mean_j = means.at<float>(j,0);
            
            const float* ptrXi = x.ptr<float>(i);
            const float* ptrXj = x.ptr<float>(j);
            
            float coeff = 0;
            for(int k=0; k<D; k++){
                coeff+=ptrXi[k]*ptrXj[k];
            }
            coeff/=D;
            coeff-=(mean_i*mean_j);
            Cxx.at<float>(i,j) = coeff;
        }
    }
    return Cxx;
}

static Mat whitening(const Mat& x){
    //== find Covariance Matrix ,ok! ==
    Mat Cxx = myCov(x);
    //======  find V = A^-1/2*OT ====
    //== find eigen value ==
    Mat eigenValues;
    Mat eigenVectors;

    eigen(Cxx, eigenValues, eigenVectors); // !! eigenVector is a row
    divide(1, eigenValues, eigenValues);
    sqrt(eigenValues, eigenValues);
    
    Mat A = Mat::diag(eigenValues); // make a diagonale matrix from the eigenValues
    //cout<<"A"<<A<<endl;
    Mat V = A*eigenVectors; // whitening matrix
    //cout<<"V"<<V<<endl;
    
    return V;
}

