#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include <vector>

/* getEigenFace - Obtiene {numComp} componentes de la matriz de vectores propios {input} con la matriz {A} que son imagenes normalizadas */
cv::Mat getEigenFace(cv::Mat eigenVec, cv::Mat A, unsigned int numComp);

/* bruteForceEigen - Carga la matriz de vectores propios usando la matrix de A*AT [2500 x 2500] */
void bruteForceEigen(cv::Mat covMat);

/* Convierte un conjunto de imagenes {image_set} en una sola matriz donde cada columna es un vector de (N^2)x1 */
cv::Mat set2matrix(std::vector<cv::Mat> &image_set);

/* Toma una proyeccion y la compara con las imagenes del set para encontrar el mejor match */
double classify(cv::Mat trainProjection, cv::Mat testProjection, std::vector<int> true_index, int setID, std::vector<int> &TP_index);