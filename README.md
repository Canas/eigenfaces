Eigenfaces
==========
This code implements the Eigenfaces algorithm in C++ using OpenCV 2.4.9. The code was originally developed as a homework for a Computer Vision course in 2014 so it possesses no optimizations whatsoever and focused solely on working as intended.

Notes
-----
- The code is intended for use with the images inside the data folder. This data contains samples of the Yale Face Database B (10 subjects, 1 pose, 64 illumination conditions).
- The eigenfaces output is a JPG file called "all_reconstructed_dX" where X is the number of components evaluated in the program.
- There are other numerous output files including yml represantions of different steps in the code. The steps use the same notation as the references and are explained in depth in the comments which are for the moment in spanish (someday I will translate all).

Instructions
------------
After compiling, the program has two ways of executing:
```bash
./eigenfaces
```
```bash
./eigenfaces (number of principal components)
```
The first one uses a default value (10) of principal componentes for the algorithm, and the second one allows the use of different numbers of components to evaluate differences in representation.

References
----------
- Eigenfaces Algorithm: http://www.scholarpedia.org/article/Eigenfaces
- Yale Face Database B: http://www.computervisiononline.com/dataset/yale-face-database-b

Contact
-------
Cristobal Silva

crsilva at ing dot uchile dot cl
