#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include <string.h>
    #include<sstream>
    


#define SET1_SIZE 70
#define SET2_SIZE 120
#define SET3_SIZE 120
#define SET4_SIZE 140
#define SET5_SIZE 190

using namespace std;
using namespace cv;

#ifdef __linux
template <typename T>
std::string to_string(T value){
	//create an output string stream
	std::ostringstream os ;
	//throw the value into the string stream
	os << value ;
	//convert the string stream into a string and return
	return os.str() ;
}
#endif

/* getImage - Carga la imagen {num} de la persona {id} */
string getImage(int id, int num){
	if (id < 10){
		if (num < 10)
			return "faces/person0" + to_string(id) + "_0" + to_string(num) + ".png";
		else
			return "faces/person0" + to_string(id) + "_" + to_string(num) + ".png";
	}
	else{
		if (num < 10)
			return "faces/person" + to_string(id) + "_0" + to_string(num) + ".png";
		else
			return "faces/person" + to_string(id) + "_" + to_string(num) + ".png";
	}
}

/* loadImage - Permite cargar imagenes iterativamente */
Mat loadImage(int &id, int &num, bool &init, int i, int setID){
	Mat output;

	int numFloor, numCeil, set_size;

	switch (setID){
	case 1:
		numFloor = 0; numCeil = 7;
		set_size = SET1_SIZE;
		break;
	case 2:
		numFloor = 7; numCeil = 19;
		set_size = SET2_SIZE;
		break;
	case 3:
		numFloor = 19; numCeil = 31;
		set_size = SET3_SIZE;
		break;
	case 4:
		numFloor = 31; numCeil = 45;
		set_size = SET4_SIZE;
		break;
	case 5:
		numFloor = 45; numCeil = 64;
		set_size = SET5_SIZE;
		break;
	default:
		cout << "Debe escoger Sets entre 1 y 5" << endl;
		return output;
	}

	if (!init){ num = numFloor; init = true; }

	if (id < 10){
		if (num < numCeil && num >= numFloor){
			output = imread(getImage(id + 1, num + 1), 0);
			//cout << "Imagen[" + to_string(i) + "] = " + getImage(id + 1, num + 1) << endl;
			num++;
		}
		else{
			num = numFloor;
			id++;

			output = imread(getImage(id + 1, num + 1), 0);
			//cout << "Imagen[" + to_string(i) + "] = " + getImage(id + 1, num + 1) << endl;
			num++;
		}
	}
	else{
		id = 0;

		if (num < numCeil && num >= numFloor){
			output = imread(getImage(id + 1, num + 1), 0);
			//cout << "Imagen[" + to_string(i) + "] = " + getImage(id + 1, num + 1) << endl;
			num++;
		}
		else{
			num = numFloor;
			id++;

			output = imread(getImage(id + 1, num + 1), 0);
			//cout << "Imagen[" + to_string(i) + "] = " + getImage(id + 1, num + 1) << endl;
			num++;
		}
	}
	return output;
}

/* writeFile - Escribe archivos .yml con nombre {dataName}  y matriz {data}*/
void writeFile(Mat data, string dataName){
	FileStorage f(dataName + ".yml", FileStorage::WRITE);
	f << dataName << data; f.release();
}

/* loadSet - Carga el Set de imagenes {setID} */
vector<Mat> loadSet(int setID){
	vector<Mat> image_set;
	int set_size;

	switch (setID){
	case 1: set_size = SET1_SIZE; break;
	case 2: set_size = SET2_SIZE; break;
	case 3: set_size = SET3_SIZE; break;
	case 4: set_size = SET4_SIZE; break;
	case 5: set_size = SET5_SIZE; break;
	}

	int id = 0; int num = 0; bool init = false;
	for (int i = 0; i < set_size; i++)
		image_set.push_back(loadImage(id, num, init, i, setID));

	return image_set;
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

int main(){
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
		imwrite("train_vectors.jpg", train_vectors);
		writeFile(train_vectors, "train_vectors");

	//------------------------------------------------------PARTE 2: OBTENER VECTOR DE MEDIAS
	Mat mean_vector(2500, 1, CV_32FC1);						// 2500 x 1
	reduce(train_vectors, mean_vector, 1, CV_REDUCE_AVG);	// 2500 x 1

		imwrite("mean_vector.jpg", mean_vector); 
		imwrite("mean_face.jpg"  , mean_vector.reshape(0, 50));
		writeFile(mean_vector, "mean_vector");

	Mat mean_matrix;
	repeat(mean_vector, 1, 70, mean_matrix);	// 2500 x 70 ---- Esto es para hacer la resta matricial

		imwrite("mean_matrix.jpg", mean_matrix);
		writeFile(mean_matrix, "mean_matrix");

	//------------------------------------------------------PARTE 3: RESTAR MATRIZ DE MEDIA A MATRIZ DE IMAGENES
	Mat L(2500, 70, CV_32FC1); // 2500 x 70
	L = train_vectors - mean_matrix;

		Mat train_reconstruct = L + mean_matrix;
		writeFile(L, "L");
		imwrite("L.jpg", L);
		imwrite("train_reconstruct.jpg", train_reconstruct);

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

	const int d = 10;							// d componentes principales
	Mat eigFaceT = getEigenFace(eigVec, L, d);	// d x 2500

		writeFile(eigFaceT, "eigFaceT");

		vector<Mat> eigFaceT_2d;
		Mat all_eigenFaces_mat;
		for (int i = 0; i < d; i++){
			Mat eigFaceT_2d_normalized;
			eigFaceT_2d.push_back((eigFaceT.row(i)).reshape(0, 50));
			normalize(eigFaceT_2d[i], eigFaceT_2d_normalized, 255, 0, NORM_MINMAX, CV_8U);
			//imwrite("eigenface_normalized" + to_string(i+1)+".jpg", eigFaceT_2d_normalized);
			imshow("eigenface " + to_string(i + 1), eigFaceT_2d_normalized);
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
			//imwrite("person" + to_string(index) + "_01_false_d" + to_string(d) + ".jpg", im_2d);
		}
		else{
			//imwrite("person" + to_string(index) + "_01_true_d" + to_string(d) + ".jpg", im_2d);
		}
	}

	Mat all_images_mat;
	hconcat(all_images, all_images_mat);
	imwrite("all_reconstructed_d" + to_string(d) + ".jpg", all_images_mat);

	cvWaitKey(0);
	return 0;
}