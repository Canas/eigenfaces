#include "file_functions.h"
#include "set_definitions.h"

using namespace std;
using namespace cv;

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
	FileStorage f("output/" + dataName + ".yml", FileStorage::WRITE);
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