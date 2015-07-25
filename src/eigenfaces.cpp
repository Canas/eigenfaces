#include "eigenfaces.h"
#include "file_functions.h"
#include "set_definitions.h"

using namespace cv;
using namespace std;

/* getEigenFace - Obtiene {numComp} componentes de la matriz de vectores propios {input} con la matriz {A} que son imagenes normalizadas */
Mat getEigenFace(Mat eigenVec, Mat A, unsigned int numComp){
	Mat aux1(70, numComp, eigenVec.type());
	Mat	aux2(2500, 1, eigenVec.type());
	Mat aux3(2500, numComp, eigenVec.type());
	Mat output(numComp, 2500, eigenVec.type());

	for (unsigned int i = 0; i < numComp; i++){
		//cout << input.row(i) << endl << endl;
		transpose(eigenVec.row(i), aux1.col(i)); 
		//cout << aux1.col(i) << endl << endl;
		
		aux2 = A*aux1.col(i);
		//cout << aux2 << endl << endl;
		(aux2).copyTo(aux3.col(i));
		//cout << aux3.col(i) << endl << endl;
		//normalize(aux2, aux3.col(i));
	}
	transpose(aux3, output);
	//cout << output << endl << endl;
	return output;
}

/* bruteForceEigen - Carga la matriz de vectores propios usando la matrix de A*AT [2500 x 2500] */
void bruteForceEigen(Mat covMat){
	Mat eigVal_brute, eigVec_bruteT;
	eigen(covMat, eigVal_brute, eigVec_bruteT);
	writeFile(eigVec_bruteT, "eigVec_bruteT");
	writeFile(eigVal_brute, "eigVal_brute");

	Mat eigenFace_bruteT(5, 2500, CV_32FC1);
	int p = 5; // p componentes
	for (int i = 0; i < p; i++){
		(eigVec_bruteT.row(i)).copyTo(eigenFace_bruteT.row(i));
	}

	writeFile(eigenFace_bruteT, "eigenFace_bruteT");

	vector<Mat> eigFaceT2d;
	for (int i = 0; i < p; i++){
		eigFaceT2d.push_back((eigenFace_bruteT.row(i)).reshape(0, 50));
		imwrite("EigenFaceBrute" + to_string(i + 1) + ".jpg", eigFaceT2d[i]);
	}
}

/* Convierte un conjunto de imagenes {image_set} en una sola matriz donde cada columna es un vector de (N^2)x1 */
Mat set2matrix(vector<Mat> &image_set){
	Mat output(image_set[0].cols*image_set[0].rows, image_set.size(), CV_32FC1);
	Mat outputT(image_set.size(), image_set[0].cols*image_set[0].rows, CV_32FC1);
	for (unsigned int i = 0; i < image_set.size(); i++)
		(image_set[i].reshape(0, 1)).copyTo(outputT.row(i));
	transpose(outputT, output);
	return output;
}

/* Toma una proyeccion y la compara con las imagenes del set para encontrar el mejor match */
double classify(Mat trainProjection, Mat testProjection, vector<int> true_index, int setID, vector<int> &TP_index){

	int test_size;
	int train_size = SET1_SIZE; // Cambiar si se cambia conjunto de entrenamiento
	vector<int> face;

	switch (setID){
	case 1: face.resize(SET1_SIZE); test_size = SET1_SIZE; break;
	case 2: face.resize(SET2_SIZE); test_size = SET2_SIZE; break;
	case 3: face.resize(SET3_SIZE); test_size = SET3_SIZE; break;
	case 4: face.resize(SET4_SIZE); test_size = SET4_SIZE; break;
	case 5: face.resize(SET5_SIZE); test_size = SET5_SIZE; break;
	}

	double min_dist = 999999;
	int fails = 0;
	int wins = 0;

	double dist;
	int index = -1;

	for (int i = 0; i < test_size; i++){
		Mat curr_image = testProjection.col(i);
		for (int j = 0; j < train_size; j++){
			dist = norm(curr_image, trainProjection.col(j), NORM_L2);
			if (dist < min_dist){
				min_dist = dist;
				index = j;
			}
		}
		face[i] = (int)floor(index / 7) + 1;
		if (face[i] != true_index[i]){
			fails++; TP_index.push_back(-1);
		}
		else{
			wins++; TP_index.push_back(face[i]);
		}
		min_dist = 999999;
	}

	double err_rate = (double)fails / (double)test_size;
	return err_rate;
}