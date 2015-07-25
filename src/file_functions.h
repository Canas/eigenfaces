#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include <iostream>
#include <string.h>
#include <sstream>

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
std::string getImage(int id, int num);

/* loadImage - Permite cargar imagenes iterativamente */
cv::Mat loadImage(int &id, int &num, bool &init, int i, int setID);

/* writeFile - Escribe archivos .yml con nombre {dataName}  y matriz {data}*/
void writeFile(cv::Mat data, std::string dataName);

/* loadSet - Carga el Set de imagenes {setID} */
std::vector<cv::Mat> loadSet(int setID);