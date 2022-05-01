// USM
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core_c.h>
#include <stack>
#include <Python.h>
#include <fstream>

#pragma warning (disable: 4996)

using namespace std;
using namespace cv;

ofstream ofs;


class CrackInfo
{
	friend std::ostream& operator << (std::ostream&, const CrackInfo&);
public:
	CrackInfo() {}
	~CrackInfo() {}
	CrackInfo(cv::Point& position, long length, float width) :Position(position), Length(length), Width(width) {}
	Point Position;

private:
	long Length;
	float Width;
	int X;
	int Y;
};

Mat src;			// 原始图像的灰度图
Mat srcCopy;		// 原始图像灰度图复制图像
Mat Gablur;			// （高斯）低频滤波
Mat UnshapeMask;	// 反锐化掩膜
Mat UsmSrc;			// src + k * UnshapeMask
Mat Usm_Binary;		// UsmSrc的二值化图像
Mat OpenMat;		// 开运算
Mat CloseMat;		// 闭运算
Mat result;			// 闭运算叠加结果
int ConnectMat;		// 联通区域
// connectedComponentsWithStats的参数
Mat labels;
Mat states;
Mat centroids;
// connectedComponentsWithStats的参数

vector<vector<Point>> connectedDomains;
vector<vector<Point>> Domains;

int n = 1;
int n_Max = 100;
int k = 1;
int k_Max = 300;
int ks = 1;
int ks_Max = 300;
int B = 1;
int B_Max = 255;
int O;
int O_Max = 100;
int NumS;
bool flag = true;
bool py_flag = true;

/*
 * 计算宽高信息的放置位置
 */
Point calInfoPosition(int imgRows, int imgCols, int padding, const vector<Point>& domain) {
	long xSum = 0;
	long ySum = 0;
	for (auto it = domain.cbegin(); it != domain.cend(); ++it) {
		xSum += it->x;
		ySum += it->y;
	}
	int x = 0;
	int y = 0;
	x = (int)(xSum / domain.size());
	y = (int)(ySum / domain.size());
	if (x < padding)
		x = padding;
	if (x > imgCols - padding)
		x = imgCols - padding;
	if (y < padding)
		y = padding;
	if (y > imgRows - padding)
		y = imgRows - padding;

	return Point(x, y);
}

/*
 * 获取图像中白点的数量
 */
void getWhitePoints(Mat& srcImg, vector<Point>& domain) {
	domain.clear();
	Mat_<uchar> tempImg = (Mat_<uchar> &)srcImg;
	for (int i = 0; i < tempImg.rows; i++) {
		uchar* row = tempImg.ptr<uchar>(i);
		for (int j = 0; j < tempImg.cols; ++j) {
			if (row[j] != 0)
				domain.push_back(Point(j, i));
		}
	}
}

/*
 * 获取连通域的骨架
 */
void thinImage(Mat& srcImg) {
	vector<Point> deleteList;
	int neighbourhood[9];
	int nl = srcImg.rows;
	int nc = srcImg.cols;
	bool inOddIterations = true;
	while (true) {
		for (int j = 1; j < (nl - 1); j++) {
			uchar* data_last = srcImg.ptr<uchar>(j - 1);
			uchar* data = srcImg.ptr<uchar>(j);
			uchar* data_next = srcImg.ptr<uchar>(j + 1);
			for (int i = 1; i < (nc - 1); i++) {
				if (data[i] == 255) {
					int whitePointCount = 0;
					neighbourhood[0] = 1;
					if (data_last[i] == 255) neighbourhood[1] = 1;
					else  neighbourhood[1] = 0;
					if (data_last[i + 1] == 255) neighbourhood[2] = 1;
					else  neighbourhood[2] = 0;
					if (data[i + 1] == 255) neighbourhood[3] = 1;
					else  neighbourhood[3] = 0;
					if (data_next[i + 1] == 255) neighbourhood[4] = 1;
					else  neighbourhood[4] = 0;
					if (data_next[i] == 255) neighbourhood[5] = 1;
					else  neighbourhood[5] = 0;
					if (data_next[i - 1] == 255) neighbourhood[6] = 1;
					else  neighbourhood[6] = 0;
					if (data[i - 1] == 255) neighbourhood[7] = 1;
					else  neighbourhood[7] = 0;
					if (data_last[i - 1] == 255) neighbourhood[8] = 1;
					else  neighbourhood[8] = 0;
					for (int k = 1; k < 9; k++) {
						whitePointCount = whitePointCount + neighbourhood[k];
					}
					if ((whitePointCount >= 2) && (whitePointCount <= 6)) {
						int ap = 0;
						if ((neighbourhood[1] == 0) && (neighbourhood[2] == 1)) ap++;
						if ((neighbourhood[2] == 0) && (neighbourhood[3] == 1)) ap++;
						if ((neighbourhood[3] == 0) && (neighbourhood[4] == 1)) ap++;
						if ((neighbourhood[4] == 0) && (neighbourhood[5] == 1)) ap++;
						if ((neighbourhood[5] == 0) && (neighbourhood[6] == 1)) ap++;
						if ((neighbourhood[6] == 0) && (neighbourhood[7] == 1)) ap++;
						if ((neighbourhood[7] == 0) && (neighbourhood[8] == 1)) ap++;
						if ((neighbourhood[8] == 0) && (neighbourhood[1] == 1)) ap++;
						if (ap == 1) {
							if (inOddIterations && (neighbourhood[3] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[5] == 0)) {
								deleteList.push_back(Point(i, j));
							}
							else if (!inOddIterations && (neighbourhood[1] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[7] == 0)) {
								deleteList.push_back(Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deleteList.size() == 0)
			break;
		for (size_t i = 0; i < deleteList.size(); i++) {
			Point tem;
			tem = deleteList[i];
			uchar* data = srcImg.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deleteList.clear();

		inOddIterations = !inOddIterations;
	}
}

/*
 * 检测连通域，并删除不符合条件的连通域
 */
 /*
  * 二值化图像。0->0,非0->255
  */
void binaryzation(Mat& srcImg)
{
	Mat lookUpTable(1, 256, CV_8U, Scalar(255));
	lookUpTable.data[0] = 0;
	LUT(srcImg, lookUpTable, srcImg);
}

//void f_Domains(Mat& srcImg, vector<vector<Point>>& Domains)
//{
//	Mat_<uchar> tempImg = (Mat_<uchar> &)srcImg;
//
//	for (int i = 0; i < tempImg.rows; ++i)
//	{
//		uchar* row = tempImg.ptr(i);
//		for (int j = 0; j < tempImg.cols; ++j)
//		{
//			stack<Point> DomainPoints;
//			vector<Point> domain;
//			DomainPoints.push(Point(j, i));
//
//			RotatedRect rect = minAreaRect(domain);
//			float width = rect.size.width;
//			float height = rect.size.height;
//			if (width < height)
//			{
//				float temp = width;
//				width = height;
//				height = temp;
//			}
//			cout << width << ", " << height << endl;
//			Domains.push_back(domain);
//
//		}
//	}
//}

int findConnectedDomain(Mat& srcImg, vector<vector<Point>>& connectedDomains, int area, int WHRatio)
{
	Mat_<uchar> tempImg = (Mat_<uchar> &)srcImg;
	int n = 0;
	for (int i = 0; i < tempImg.rows; ++i)
	{
		uchar* row = tempImg.ptr(i);
		for (int j = 0; j < tempImg.cols; ++j)
		{
			if (row[j] == 255)
			{
				stack<Point> connectedPoints;
				vector<Point> domain;
				connectedPoints.push(Point(j, i));
				while (!connectedPoints.empty())
				{
					Point currentPoint = connectedPoints.top();
					domain.push_back(currentPoint);

					int colNum = currentPoint.x;
					int rowNum = currentPoint.y;

					tempImg.ptr(rowNum)[colNum] = 0;
					connectedPoints.pop();

					if (rowNum - 1 >= 0 && colNum - 1 >= 0 && tempImg.ptr(rowNum - 1)[colNum - 1] == 255)
					{
						tempImg.ptr(rowNum - 1)[colNum - 1] = 0;
						connectedPoints.push(Point(colNum - 1, rowNum - 1));
					}
					if (rowNum - 1 >= 0 && tempImg.ptr(rowNum - 1)[colNum] == 255)
					{
						tempImg.ptr(rowNum - 1)[colNum] = 0;
						connectedPoints.push(Point(colNum, rowNum - 1));
					}
					if (rowNum - 1 >= 0 && colNum + 1 < tempImg.cols && tempImg.ptr(rowNum - 1)[colNum + 1] == 255)
					{
						tempImg.ptr(rowNum - 1)[colNum + 1] = 0;
						connectedPoints.push(Point(colNum + 1, rowNum - 1));
					}
					if (colNum - 1 >= 0 && tempImg.ptr(rowNum)[colNum - 1] == 255)
					{
						tempImg.ptr(rowNum)[colNum - 1] = 0;
						connectedPoints.push(Point(colNum - 1, rowNum));
					}
					if (colNum + 1 < tempImg.cols && tempImg.ptr(rowNum)[colNum + 1] == 255)
					{
						tempImg.ptr(rowNum)[colNum + 1] = 0;
						connectedPoints.push(Point(colNum + 1, rowNum));
					}
					if (rowNum + 1 < tempImg.rows && colNum - 1 > 0 && tempImg.ptr(rowNum + 1)[colNum - 1] == 255)
					{
						tempImg.ptr(rowNum + 1)[colNum - 1] = 0;
						connectedPoints.push(Point(colNum - 1, rowNum + 1));
					}
					if (rowNum + 1 < tempImg.rows && tempImg.ptr(rowNum + 1)[colNum] == 255)
					{
						tempImg.ptr(rowNum + 1)[colNum] = 0;
						connectedPoints.push(Point(colNum, rowNum + 1));
					}
					if (rowNum + 1 < tempImg.rows && colNum + 1 < tempImg.cols && tempImg.ptr(rowNum + 1)[colNum + 1] == 255)
					{
						tempImg.ptr(rowNum + 1)[colNum + 1] = 0;
						connectedPoints.push(Point(colNum + 1, rowNum + 1));
					}
				}
				//cout << (int)srcImg.ptr(29)[301] << endl;
				NumS = 0;
				double buffer[5];

				if (domain.size() > area)
				{
					RotatedRect rect = minAreaRect(domain);
					float width = rect.size.width;
					float height = rect.size.height;
					if (width < height)
					{
						float temp = width;
						width = height;
						height = temp;
					}
					if (width > height * WHRatio && width > 5)
					{
						long int Zhi = 0;
						int m = 0;
						auto cit = domain.begin();
						string Num = to_string(n);
						putText(srcImg, Num, Point(cit->x, cit->y), FONT_HERSHEY_COMPLEX, 1, Scalar(250, 255, 100));
						for (; cit != domain.end(); ++cit)
						{
							m++;
							Zhi += (int)srcCopy.ptr(cit->y)[cit->x];
							tempImg.ptr(cit->y)[cit->x] = 250;
						}
						//cout << Zhi << "\t" << m << endl;
						double Zhi1;
						Zhi1 = Zhi / 255.0 / m;
						if (n != 0)
						{
							Zhi1 /= n;
						}
						double fancha = 0.0;
						for (auto cits = domain.begin(); cits != domain.end(); ++cits)
						{
							double Zhi2 = (int)srcCopy.ptr(cits->y)[cits->x] / 255.0 - Zhi1;
							fancha += pow(Zhi2, 2);
						}
						fancha /= m;
						fancha = sqrt(fancha);
						//printf("%.6f\t%.6f\t%.6f\t%.6f\t%0.6f\t%d\n", width, height, (float)domain.size(), Zhi1, fancha, n);
						//printf("%.6f\t%.6f\t%.6f\t%.6f\t%0.6f\t0\n", width, height, (float)domain.size(), Zhi1, fancha);
						//printf("%.6f\t%.6f\t%.6f\t%.6f\t%0.6f\n", width, height, (float)domain.size(), Zhi1, fancha);
						int i = 0;
						if (flag == true)
						{
							ofs.open("Test_Data.txt", ios::app);
							//cout << ofs.is_open() << endl;
							ofs << width << "\t" << height << "\t" << (float)domain.size() << "\t" << Zhi1 << "\t" << fancha << endl;
							i++;
							ofs.close();
						}
						//cout << n << "\t" << width << "\t" << height << "\t" << domain.size() << endl;
						n++;
						//putText(UsmSrc, p[i], Point(point_it->x, point_it->y), FONT_HERSHEY_COMPLEX, 1, Scalar(250, 255, 100));
						connectedDomains.push_back(domain);
					}
				}
			}
		}
	}
	return NumS;
	//threshold(UsmSrc, UsmSrc, 240, 255, THRESH_BINARY_INV);
	//binaryzation(srcImg);
}


/*
 * 增强图像对比度
 */
void addConstrast(Mat& srcImg)
{
	Mat lookUpTable(1, 256, CV_8U);
	double temp = pow(1.1, 5);
	uchar* p = lookUpTable.data;
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(i * temp);
	LUT(srcImg, lookUpTable, srcImg);
}

Point Loac = Point(0, 0);
/*
 * 鼠标回显操作
 *
 * 在UnshapeMask画出在原图中点击的点
 *
 */
void onMouse(int event, int x, int y, int flag, void* param)
{
	Mat* im = reinterpret_cast<Mat*>(param);
	if (event == EVENT_LBUTTONDOWN)
	{
		Loac = Point(x, y);
		int Zhi = (int)UnshapeMask.ptr(y)[x];
		float Zhi1 = Zhi / 255.0;
		cout << "at(" << x << ", " << y << ")value is:" << Point(x, y) << ", " << Zhi1 << endl;
		circle(src, Point(x, y), 1, Scalar(0, 0, 0), 1);
		circle(UnshapeMask, Point(x, y), 1, Scalar(255, 255, 255), 1);
		//imshow("灰度图", src);
		//imshow("UnshapeMask", UnshapeMask);
	}
}
void click_on_the_image(Mat image)
{
	setMouseCallback("灰度图", onMouse, reinterpret_cast<void*> (&image));
}

/*
 * 调用PyThon
 */
void PythonFun()
{
	Py_Initialize();

	if (!Py_IsInitialized())
	{
		cout << "初始化失败！" << endl;
	}

	PyObject* pModule = NULL;
	PyObject* pFunc1 = NULL;

	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('src/data/')");

	pModule = PyImport_ImportModule("PNN");

	if (pModule == NULL)
	{
		cout << "没找到" << endl;
	}
	else
	{
		pFunc1 = PyObject_GetAttrString(pModule, "main");
		PyEval_CallObject(pFunc1, NULL);
		Py_Finalize();
	}
}

/*
 * 滑块函数
 *
 * 包含操作：
 *		1. 高斯低频滤波
 *		2. 原图像减去低频部分得到反锐化掩膜
 *		3. 原始图像加上K倍反锐化掩膜
 *		4. 对反锐化掩膜图像进行非线性增益
 *		5. 原始图像加上非线性增益处理后的反锐化掩膜
 *
 */
void operate(int, void*)
{
	// 高斯低频滤波
	//GaussianBlur(src, Gablur, Size(2 * n + 1, 2 * n + 1), 0);
	GaussianBlur(src, Gablur, Size(15, 15), 0);
	//imshow("Gablur", Gablur);
	// 原图像减去低频部分得到反锐化掩膜
	//UnshapeMask = src - Gablur;
	subtract(Gablur, src, UnshapeMask);
	/*imshow*/("UnshapeMask", UnshapeMask);

	addWeighted(src, 1, UnshapeMask, k, 0, UsmSrc);
	/*imshow*/("UsmSrc", UsmSrc);

	// 对反锐化掩膜图像进行非线性增益
	for (int i = 0; i < UnshapeMask.cols; i++)
	{
		for (int j = 0; j < UnshapeMask.rows; j++)
		{
			int Zhi = (int)UnshapeMask.ptr(j)[i];
			double Zhi1 = Zhi / 255.0;
			if (Zhi1 > 0.01 && Zhi1 < 0.08)
			{
				UnshapeMask.ptr(j)[i] = 35 * UnshapeMask.ptr(j)[i];
			}
			else if (Zhi1 > 0.1)
			{
				UnshapeMask.ptr(j)[i] = -5 * UnshapeMask.ptr(j)[i];
			}
			else
			{
				UnshapeMask.ptr(j)[i] = 0;
			}
		}
	}

	addWeighted(src, 1, UnshapeMask, 10, 0, UsmSrc);
	//addWeighted(src, 1, UnshapeMask, ks = 10, 0, UsmSrc);
	/*imshow*/("UsmSrc1", UsmSrc);

	//threshold(UsmSrc, Usm_Binary, B = 155, 255, THRESH_BINARY_INV);
	//imshow("二值化图像", Usm_Binary);

	// Test

	/*
	 *
	 * 做开、闭运算
	 *
	 */
	 //Mat element = getStructuringElement(MORPH_OPEN, Size(1, 1));
	 //morphologyEx(Usm_Binary, OpenMat, MORPH_OPEN, element);		// 开运算
	 //Mat element1 = getStructuringElement(MORPH_OPEN, Size(1, 1));
	 //morphologyEx(OpenMat, CloseMat, MORPH_CLOSE, element1);		// 闭运算
	 ////颜色反转
	 //for (int i = 0; i < CloseMat.cols; i++)
	 //	for (int j = 0; j < CloseMat.rows; j++)
	 //		CloseMat.ptr(j)[i] = 255 - CloseMat.ptr(j)[i];
	 //addWeighted(src, 1, CloseMat, 1, 0, result);
	 //imshow("叠加结果", result);


	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	dilate(UsmSrc, UsmSrc, kernel);
	morphologyEx(UsmSrc, UsmSrc, MORPH_CLOSE, kernel, Point(-1, -1), 3);
	morphologyEx(UsmSrc, UsmSrc, MORPH_CLOSE, kernel);
	/*imshow*/("src", UsmSrc);

	int Number = findConnectedDomain(UsmSrc, connectedDomains, 300, 1);
	flag = false;
	/*imshow*/("src1", UsmSrc);
	kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(UsmSrc, UsmSrc, MORPH_CLOSE, kernel, Point(-1, -1), 2);
	/*imshow*/("src1s", UsmSrc);
	//cout << connectedDomains.size() << endl;

	//connectedDomains.clear();
	findConnectedDomain(UsmSrc, connectedDomains, 300, 3);
	kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
	morphologyEx(UsmSrc, UsmSrc, MORPH_OPEN, kernel);
	/*imshow*/("src2", UsmSrc);

	kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	erode(UsmSrc, UsmSrc, kernel);
	/*imshow*/("src3", UsmSrc);
	//cout << connectedDomains.size() << endl;


	//connectedDomains.clear();
	findConnectedDomain(UsmSrc, connectedDomains, 300, 3);
	/*imshow*/("src4", UsmSrc);

	Mat lookUpTable(1, 256, CV_8U, Scalar(0));
	vector<CrackInfo> crackInfos;
	for (auto domain_it = connectedDomains.begin(); domain_it != connectedDomains.end(); ++domain_it)
	{
		LUT(UsmSrc, lookUpTable, UsmSrc);
		for (auto point_it = domain_it->cbegin(); point_it != domain_it->cend(); ++point_it)
		{
			UsmSrc.ptr<uchar>(point_it->y)[point_it->x] = 255;
		}
		double area = (double)domain_it->size();
		thinImage(UsmSrc);
		getWhitePoints(UsmSrc, *domain_it);
		long length = (long)domain_it->size();
		Point position = calInfoPosition(UsmSrc.rows, UsmSrc.cols, 50, *domain_it);
		crackInfos.push_back(CrackInfo(position, length, (float)(area / length)));
	}

	LUT(UsmSrc, lookUpTable, UsmSrc);

	//cout << connectedDomains.size() << endl;
	int n = 0;
	const int N = connectedDomains.size();
	string* p = new string[1000000];
	for (auto domain_it = connectedDomains.cbegin(); domain_it != connectedDomains.cend(); ++domain_it)
	{
		n++;
		string Num = to_string(n);
		//cout << Num << "***";
		p[n] = Num;
	}
	//cout << n << endl;
	auto domain_it = connectedDomains.cbegin();
	for (auto domain_it = connectedDomains.cbegin(); domain_it != connectedDomains.cend(); ++domain_it)
	{
		//cout << width << ", " << height << ", " << domain.size() << endl;
		//cout << domain_it->size() << endl;
		auto point_it = domain_it->cbegin();
		//putText(UsmSrc, p[i], Point(point_it->x, point_it->y), FONT_HERSHEY_COMPLEX, 1, Scalar(250, 255, 100));
		for (; point_it != domain_it->cend(); ++point_it)
		{
			UsmSrc.ptr<uchar>(point_it->y)[point_it->x] = 255;
		}
	}

	/*imshow*/("src5", UsmSrc);

	if (py_flag)
	{
		cout << "执行" << endl;
		PythonFun();
		py_flag = false;
	}

	click_on_the_image(src);

}

int main()
{
	string srcPath;
	cout << "请输入图片路径：" << endl;
	// Source/Normal1.1.png --好蛋
	// Source/block1.1.png --坏蛋
	cin >> srcPath;
	src = imread(srcPath, CV_8UC1);
	namedWindow("原图", WINDOW_NORMAL);
	namedWindow("灰度图", WINDOW_GUI_NORMAL);
	imshow("原图", src);
	src.copyTo(srcCopy);
	//addConstrast(src);

	// 阈值自适应
	//adaptiveThreshold(src, src, 255, 0, 1, 5, 10);
	//imshow("灰度图", src);

	namedWindow("UnshapeMask", WINDOW_NORMAL);
	namedWindow("UsmSrc", WINDOW_NORMAL);
	namedWindow("UsmSrc1", WINDOW_NORMAL);
	namedWindow("二值化图像", WINDOW_NORMAL);
	//namedWindow("开运算结果", WINDOW_NORMAL);
	//namedWindow("叠加结果", WINDOW_NORMAL);
	//namedWindow("原图", WINDOW_NORMAL);

	createTrackbar("半径大小", "UnshapeMask", &n, n_Max, operate);
	createTrackbar("K值大小", "UsmSrc", &k, k_Max, operate);
	createTrackbar("Ks值大小", "UsmSrc1", &ks, ks_Max, operate);
	createTrackbar("阈值", "二值化图像", &B, B_Max, operate);

	waitKey(0);
	return 0;
}