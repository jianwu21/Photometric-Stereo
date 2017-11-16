Photometric-Stereo
==================

Implementation
--------------
There are a couple of assumptions made for the datasets, we are assuming the
materials are perfectly matte (Lambertian), the pictures are all from the same
camera angle but with varying light angles. According to Lambert’s law, for a given pixel at [u, v],

<img src='/src/1.png' width='400'>

Where I is the vector of k observed intensities, S is the known 3 x k matrix of normalized light directions, and M is the surface normal that we need. The
algorithms is then for each (valid) pixel [u, v] in image domain, solve m(u, v)
via Moore-Penrose pseudoinverse or equation selection. Then we get the albedo and normal with the following equation:

<img src='/src/2.png'>

Code
----
Please make sure your ``python`` env is ``python2.7``  
Run:
```shell
python Buddaha.py
```
and   
```shell
python Buddaha_improved.py
```

Beethoven
---------
The original image from three directions of light.

<img src='/src/Figure_1.png'>

<img src='/src/Figure_2.png' width='500'>

Then we can get the 3D image from these images.

<img src='/src/Figure_3.png' width='500'>

Buddaha
-------
<img src='/src/Figure_1_budda.png'>

<img src='/src/Figure_2_budda.png' width='500'>

<img src='/src/Figure_3_budda.png' width='500'>

Improvement
-----------
we made some improvements for the Buddha dataset. We found out that using all
10 observations in the dataset with Penrose inverse doesn’t provide a
satisfying result, and decided to use equation selection with 3 good
measurements selected out of 10. We first tried the method of choosing the
image with highest de tMabc , but then the result has nan for all z at all
points, thus we abandoned this method, and decided to hand pick the images. In
order to find the final normal vector n, we must have images that have light
emitting from left, centre, and right, thus we have 3 types of images by light
angle, [0,1,2,3],[4,5],[6,7,8,9], thus 4 x 2 x 4 = 32 choices. Since finding z
value for all choices and creating a 3-d rendering is very slow, we calculated
all possible z values and produced a contour map, which is a much faster process. Eventually we chose [1,4,7] for the equation.
