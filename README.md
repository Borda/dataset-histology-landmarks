# Dataset: histology landmarks

[![Build Status](https://travis-ci.org/Borda/dataset-histology-landmarks.svg?branch=master)](https://travis-ci.org/Borda/dataset-histology-landmarks)
[![codecov](https://codecov.io/gh/Borda/dataset-histology-landmarks/branch/master/graph/badge.svg)](https://codecov.io/gh/Borda/dataset-histology-landmarks)
[![codebeat badge](https://codebeat.co/badges/6c0bfead-09bc-42ed-aa8e-cf49944aaa40)](https://codebeat.co/projects/github-com-borda-dataset-histology-landmarks-master)
[![Maintainability](https://api.codeclimate.com/v1/badges/e1374e80994253cc8e95/maintainability)](https://codeclimate.com/github/Borda/dataset-histology-landmarks/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/e1374e80994253cc8e95/test_coverage)](https://codeclimate.com/github/Borda/dataset-histology-landmarks/test_coverage)


**Dataset: landmarks for registration of [CIMA histology images](http://cmp.felk.cvut.cz/~borovji3/?page=dataset)**

The dataset consists of 2D histological microscopy tissue slices differently stained. 
The main challenges for the registration of these images are the following: very large image size, appearance differences, and lack of distinctive appearance objects. 
Our dataset contains 108 image pars and manually placed landmarks for registration quality evaluation.

![reconstruction](figures/images-landmarks.jpg)

The images composing the CIMA dataset are available [here](http://cmp.felk.cvut.cz/~borovji3/?page=dataset). 
**Note** that the available landmarks are mostly the result from a single user annotation. 
This is a work in progress. It would be interesting to have more precise landmarks computed as the fusion of several users' annotations. Please, consider to contribute to the task!

---

## Landmarks

The landmarks have standard [ImageJ](https://imagej.net/Welcome) structure and coordinate frame. 
The origin [0, 0] is located in top left corner of the image plane. 
For handling these landmarks, we provide a simple macro for [importing](annotations/multiPointSet_import.ijm) and another one for [exporting](annotations/multiPointSet_export.ijm).

The structure of the landmarks file is as follows:
```
 ,X,Y
1,226,173
2,256,171
3,278,182
4,346,207
...
```
 and it can be simply imported by `pandas` as `DataFrame`.

The landmarks files are stored in the same folder as their corresponding images and share the same name. 
```
**DATASET**
 |- [set_name1]
 |  |- scale-[number1]pc
 |  |   |- [image_name1].jpg
 |  |   |- [image_name1].csv
 |  |   |- [image_name2].jpg
 |  |   |- [image_name2].csv
 |  |   |  ...
 |  |   |- [image_name].jpg
 |  |   '- [image_name].csv
 |  |- scale-[number2]pc
 |  |  ...
 |  '- scale-[number]pc
 |      |- [image_name1].png
 |      |- [image_name1].csv
 |      |  ...
 |      |- [image_name].png
 |      '- [image_name].csv
 |- [set_name2]
 | ...
 '- [set_name]
```
See the attached [dataset](dataset) examples.

The landmarks for all the images are generated as consensus over all the available expert annotations: 
```bash
python handlers/run_generate_landmarks.py \
    -a ./annotations -d ./dataset \
    --scales 5 10 25 50
```

A routine to visualize the landmarks overlay on the histological image has been created. 
It is also possible to visualize histological images pairs and the correspondence between the landmarks pairs. 
Namely, the landmarks pairs in both images are connected by a line.  
It is expected that in this case, the landmarks' main direction of displacement can be observed and the affine transformation relating the pair of images, estimated.
```bash
python handlers/run_visualise_landmarks.py \
    -l ./dataset -i ./dataset -o ./output \
    --scales 5 10 --nb_jobs 2
```

There is a verification procedure before any new annotation is added to the "authorised" annotation. 
First, it is checked that you did not swap any landmark or disorder them. 
This can be simply observed from the main displacement direction of all the landmarks in a particular sequence of images pairs. 
Second, the error of the new annotation should not be significantly larger than that of a reference annotation. 

---

## Annotations

The annotation is a collection of landmarks placed by several users. 
The structure is similar to the one used in the dataset with the minor difference that there is user/author "name" and the annotation is made just in a single scale.

![reconstruction](figures/imagej-image-pair.jpg)

Tutorial how to put landmarks in a set of images step by step:
1. Open **Fiji**
2. Load the images. It is optimal to open the complete set. 
3. Click relevant points (landmarks) in all images.
4. Export the placed landmarks.
5. Import the existing landmarks if needed.

Structure of the annotation directory:
```
**ANNOTATIONS**
 |- [set_name1]
 |  |- user-[initials1]_scale-[number2]pc
 |  |   |- [image_name1].csv
 |  |   |- [image_name2].csv
 |  |   |  ...
 |  |   '- [image_name].csv
 |  |- user-[initials2]_scale-[number1]pc
 |  |  ...
 |  |- user-[initials]_scale-[number]pc
 |  |   |- [image_name2].csv
 |  |   |  ...
 |  |   '- [image_name].csv
 |- [set_name2]
 | ...
 '- [set_name]
```
See the attached dataset [annotation](annotations).

### Placement of relevant points

Because it is not possible to remove already placed landmarks, check if the partial structure you plan to annotate appears in all images before you place the first landmark in any image:
1. Select `Multi-point tool`, note that the points are indexed so you can verify that the actual points are fine.
2. To move in the image use Move tool and also Zoom to see the details.
3. Place landmarks in significant structures of the tissue like edges or alveolar sac centroids appearing in all cuts of the tissue. Each image should contain about 80 landmarks.

![reconstruction](figures/landmarks-zoom.jpg)

### Work with Export / Import macros

**Exporting finally placed landmarks**
When you finish to place landmarks on all the images, export each of them into separate files.
1. Install macro to export landmarks: Select `Plugins -> Marcos -> Install...`
and then, select exporting macro `annotations/multiPointSet_export.ijm`.
2. Select one image and click `Plugins -> Marcos -> exportMultipointSet`.
3. Choose for the landmark file the same name as the one of the image (without any annex).
4. The macro automatically exports all landmarks corresponding to the image in `.csv` format to the chosen directory.

**Importing existing landmarks**
When you need to correct for previously stored landmarks or to continue annotating an image, load the landmarks from a file using the importing existing landmarks macro. Note that the macro uses `.csv` format.
1. Install importing macro `annotations/multiPointSet_import.ijm`.
2. Select one image and click `Plugins -> Marcos -> importMultipointSet`.
3. Then, the correct landmarks set can be selected by its name. 


### Validation

There are to scenarios of validation - if you are the first user doing annotation for the particular tissue there is only visual inspection with highlighting larger deformation from estimated affine transform. 
```bash
python handlers/run_visualise_landmarks.py \
    -l ./annotations -i ./dataset -o ./output \
    --nb_jobs 2
```

In the visualization, the landmarks pairs in both images are connected by a line. 
An affine transformation is computed between the two sets of landmarks. 
Then, the error between the landmarks in the second image and the warped landmarks computed from the second image are computed. 
Finally, if the error is larger than five standard deviations, we consider them as a suspicious pair (either the result of a wrong localization of a large elastic deformation). 
In the visualization, they are connected by a straight line. Otherwise, the landmark pair si connected by a dotted line. 

![landmarks-pairs](figures/PAIR___29-041-Izd2-w35-CD31-3-les3___AND___29-041-Izd2-w35-proSPC-4-les3.jpg)

In case you do another annotation for tissue with already existing annotations you should also check your difference to the already existing consensus landmarks makde by other users/experts and are saved in a file.
Then, you need to focus on landmarks where the standard deviation or the maximal error value exceed a reasonable value.
```bash
python handlers/run_evaluate_landmarks.py \
    -a ./annotations -o ./output
```
If you find such suspicious annotations, perform a visual inspection as described above.

![landmarks-pairs-warped](figures/landmarks-overlaps-warped.jpg)

We recommend looking at the warped image pairs (it is generated automatically if you  have installed OpenCV) where an affine transformation between two images was estimated from landmarks and the second image with landmarks was warped to the first image.

---

## References

J. Borovec, A. Munoz-Barrutia, and J. Kybic, “**Benchmarking of image registration methods for differently stained histological slides**” in 2018 25th IEEE International Conference on Image Processing (ICIP), p. 3368-3372, 2018.
