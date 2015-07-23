Eigenfaces
==========
This code implements the Eigenfaces algorithm in C++ using OpenCV 2.4.9. The code was originally developed as a homework for a Computer Vision course in 2014 so it possesses no optimizations whatsoever and focused solely on working as intended.

Notes
-----
- The code was originally intended for use with the data available here (www.cec.uchile.cl/~crsilva/projects/eigenfaces/faces.tar.gz). This data contains samples of the Yale Face Database B (10 subjects, 1 pose, 64 illumination conditions).

Instructions
------------
After compiling, the program has two ways of executing:
```bash
./eigenfaces
```
```bash
./eigenfaces (number of principal components)
```
The first one uses a default value (7) of principal componentes for the algorithm, and the second one allows the use of more components for a more accurate representation.

References
----------
- Eigenfaces Algorithm: http://www.scholarpedia.org/article/Eigenfaces
- Yale Face Database B: http://www.computervisiononline.com/dataset/yale-face-database-b

Contact
-------
Cristobal Silva

crsilva at ing dot uchile dot cl