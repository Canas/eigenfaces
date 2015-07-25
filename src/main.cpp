#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include "file_functions.h"
#include "eigenfaces.h"
#include "set_definitions.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
	int d;
	if(argc > 1){
		d = atoi(argv[1]);
	}
	else{
		d = 10;
	}

	// CARGAR CONJUNTOS ENTRENAMIENTO Y PRUEBA
	vector<Mat> train_set = loadSet(1);
	vector< vector<Mat> > test_set;
	for (int i = 1; i <= 5; i++){
		test_set.push_back(loadSet(i));
	}

	/* ------------------ PCA ------------------*/
	//------------------------------------------------------PARTE 1: Convertir M imagenes de 50x50 en una sola matriz de [2500 x M]
	Mat train_vectors = set2matrix(train_set);
	vector<Mat> test_vectors;
	for (int i = 0; i < 5; i++){
		test_vectors.push_back(set2matrix(test_set[i]));	// 2500 x N cada set
	}
	
		writeFile(test_vectors[1], "set_2");
		imwrite("output/train_vectors.jpg", train_vectors);
		writeFile(train_vectors, "train_vectors");

	//------------------------------------------------------PARTE 2: OBTENER VECTOR DE MEDIAS
	Mat mean_vector(2500, 1, CV_32FC1);						// 2500 x 1
	reduce(train_vectors, mean_vector, 1, CV_REDUCE_AVG);	// 2500 x 1

		imwrite("output/mean_vector.jpg", mean_vector); 
		imwrite("output/mean_face.jpg"  , mean_vector.reshape(0, 50));
		writeFile(mean_vector, "mean_vector");

	Mat mean_matrix;
	repeat(mean_vector, 1, 70, mean_matrix);	// 2500 x 70 ---- Esto es para hacer la resta matricial

		imwrite("output/mean_matrix.jpg", mean_matrix);
		writeFile(mean_matrix, "mean_matrix");

	//------------------------------------------------------PARTE 3: RESTAR MATRIZ DE MEDIA A MATRIZ DE IMAGENES
	Mat L(2500, 70, CV_32FC1); // 2500 x 70
	L = train_vectors - mean_matrix;

		Mat train_reconstruct = L + mean_matrix;
		writeFile(L, "L");
		imwrite("output/L.jpg", L);
		imwrite("output/train_reconstruct.jpg", train_reconstruct);

	/*------------------PARTE 4: CALCULAR MATRIZ DE COVARIANZA covMat */
	Mat LT;	transpose(L, LT);	// 70 x 2500
	Mat covMat = L*LT;			// [2500 x 70]*[70 x 2500] = [2500 x 2500]
	Mat auxMat = LT*L;			// [70 x 2500]*[2500 x 70] = [70 x 70]
		
		writeFile(covMat, "covMat");
		writeFile(auxMat, "auxMat");
	
	//------------------------------------------------------PARTE 5: OBTENER d COMPONENTES PRINCIPALES
	Mat eigVal, eigVec;
	eigen(auxMat, eigVal, eigVec);	// Eigen sobre 70 x 70
	//bruteForceEigen(covMat);		// Eigen sobre 2500 x 2500 - DESCOMENTAR PARA USAR

		writeFile(eigVec, "eigVec");
		writeFile(eigVal, "eigVal");

	//const int d = 10;							// d componentes principales
	Mat eigFaceT = getEigenFace(eigVec, L, d);	// d x 2500

		writeFile(eigFaceT, "eigFaceT");

		vector<Mat> eigFaceT_2d;
		Mat all_eigenFaces_mat;
		for (int i = 0; i < d; i++){
			Mat eigFaceT_2d_normalized;
			eigFaceT_2d.push_back((eigFaceT.row(i)).reshape(0, 50));
			normalize(eigFaceT_2d[i], eigFaceT_2d_normalized, 255, 0, NORM_MINMAX, CV_8U);
			//imwrite("output/eigenface_normalized" + to_string(i+1)+".jpg", eigFaceT_2d_normalized);
			//imshow("eigenface " + to_string(i + 1), eigFaceT_2d_normalized);
		}


	Mat eigFaceNormT(d, 2500, CV_32FC1);
	for (int i = 0; i < eigFaceT.rows; i++){	// Normalizar la transformación v_i = A * u_i
		normalize(eigFaceT.row(i), eigFaceNormT.row(i));
	}
	
		writeFile(eigFaceNormT, "eigFaceNormT");
	
	Mat eigFaceNorm(2500, d, CV_32FC1);			// 2500 x d --- Se transpone para dejar vectores en notacion de columnas
	transpose(eigFaceNormT, eigFaceNorm);
	
		writeFile(eigFaceNorm, "eigFaceNorm");
		
	//------------------------------------------------------PARTE 6: PROYECTAR CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
	// ETIQUETAS CONJUNTO DE ENTRENAMIENTO
	vector<int> gnd_face(SET1_SIZE);
	for (unsigned int i = 0; i < gnd_face.size(); i++){ gnd_face[i] = (int)floor(i / 7) + 1; }
	// ETIQUETAS CONJUNTO DE PRUEBA
	vector< vector<int> > set_face(test_set.size()); int face_index;
	for (unsigned int i = 0; i < test_set.size(); i++){
		for (unsigned int j = 0; j < test_set[i].size(); j++){
			face_index = (int)floor(j / (test_set[i].size() / 10)) + 1;
			set_face[i].push_back(face_index);
		}
	}

	// RECONOCIMIENTO DE CARAS
	// Obtener caras de prueba sin la media
	vector<Mat> A;
	for (int i = 0; i < 5; i++){
		Mat mean_i;
		repeat(mean_vector, 1, test_set[i].size(), mean_i);
		A.push_back(test_vectors[i] - mean_i); // [2500 x N] - [2500 x N]

		if (i == 1)
			writeFile(mean_i, "mean_matrix_set2");

	}

	// Proyecciones de todos los conjuntos
	Mat gnd_t = eigFaceNormT * L;	// Vectores BASE
	Mat proj1 = eigFaceNormT * A[0];
	Mat proj2 = eigFaceNormT * A[1];
	Mat proj3 = eigFaceNormT * A[2];
	Mat proj4 = eigFaceNormT * A[3];
	Mat proj5 = eigFaceNormT * A[4];

	writeFile(proj1, "proj1"); writeFile(A[0], "A0");
	writeFile(proj2, "proj2"); writeFile(A[1], "A1");
	writeFile(proj3, "proj3");
	writeFile(proj4, "proj4");
	writeFile(proj5, "proj5");

	vector<int> TP_index1, TP_index2, TP_index3, TP_index4, TP_index5;
	vector< vector<int> > TP_index(5);
	double err1 = classify(gnd_t, proj1, set_face[0], 1, TP_index[0]);
	double err2 = classify(gnd_t, proj2, set_face[1], 2, TP_index[1]);
	double err3 = classify(gnd_t, proj3, set_face[2], 3, TP_index[2]);
	double err4 = classify(gnd_t, proj4, set_face[3], 4, TP_index[3]);
	double err5 = classify(gnd_t, proj5, set_face[4], 5, TP_index[4]);

	//ofstream file("error_data_d" + to_string(d) + ".txt");
	cout << "Numero de componentes principales: " << d << endl;
	cout << "Set 1 - Set size: " << SET1_SIZE << endl;
	cout << "Error rate: " << err1 << endl;
	cout << "Set 2 - Set size: " << SET2_SIZE << endl;
	cout << "Error rate: " << err2 << endl;
	cout << "Set 3 - Set size: " << SET3_SIZE << endl;
	cout << "Error rate: " << err3 << endl;
	cout << "Set 4 - Set size: " << SET4_SIZE << endl;
	cout << "Error rate: " << err4 << endl;
	cout << "Set 5 - Set size: " << SET5_SIZE << endl;
	cout << "Error rate: " << err5 << endl;
	//file.close();

	// RECONSTRUCCION DE CARAS DETECTADAS - Person*01.png (10 imagenes)
	Mat set1_rec(2500, 70, CV_32FC1);
	set1_rec = (eigFaceNorm*proj1);
	set1_rec = set1_rec + mean_matrix;

	Mat im1 = (set1_rec.col(0)).clone();
	Mat im1_2d = im1.reshape(0, 50);

	im1_2d.convertTo(im1_2d, CV_8UC1);
	//imshow("im1_2d", im1_2d);
	//cvWaitKey(0);

	int index;
	Mat set_rec, im, im_2d;

	vector<Mat> all_images;
	for (int i = 0; i < 70; i += 7){
		index = (int)floor(i / 7) + 1;
		set_rec = (eigFaceNorm*proj1) + mean_matrix;
		im_2d = (set_rec.col(i)).clone().reshape(0, 50);

		all_images.push_back(im_2d);

		if (TP_index[0][i] < 0){	
			//imwrite("output/person" + to_string(index) + "_01_false_d" + to_string(d) + ".jpg", im_2d);
		}
		else{
			//imwrite("output/person" + to_string(index) + "_01_true_d" + to_string(d) + ".jpg", im_2d);
		}
	}

	Mat all_images_mat;
	hconcat(all_images, all_images_mat);
	imwrite("output/all_reconstructed_d" + to_string(d) + ".jpg", all_images_mat);
	all_images_mat.convertTo(all_images_mat,CV_8U);
	imshow("eigenfaces", all_images_mat);
	cout << "Press CTRL-C or ENTER to exit program." << endl;
	cvWaitKey(0);
	return 0;
}