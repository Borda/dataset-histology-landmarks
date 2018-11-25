#!/bin/bash

# simple bash to create several scaling of full images using ImageMagick
# https://www.imagemagick.org/script/convert.php
# images has to be saved just in the folder "scale-100pc", no sub-folders
# expected folder structure is DATASET/<tissue>/scale-100pc/<image>
# where the scaled versions are saved in DATASET/<tissue>/scale-<number>pc/<image>
# >> handlers/resize_images.sh dataset/lesions_1 .jpg

# input tissue folder
input_folder=$1
# annex if converting images
img_ext=$2

# converting sizes in per-cent
size_set=(5 10 25 50)
# name of the original / full folder
base_folder="scale-100pc"
path_source=$input_folder/$base_folder
echo "source: $path_source"

for size in ${size_set[@]}; do

    scaled_folder="scale-"$size"pc"
	# path_output=$input_folder"/"$scaled_folder
	path_output=${path_source/$base_folder/$scaled_folder}
	echo "scale folder: $path_output"
	if [ ! -d $path_output ]
	  then
	    mkdir $path_output
	fi

	echo "listing in $path_source/*$img_ext"
	list_images=$path_source"/*"$img_ext
	for img in $list_images ; do

		img_name=${img##*/}
		img_name=${img_name%.*}
		cmd="convert -resize $size% -interpolate spline -define jp2:rate=1.0 $img $path_output/$img_name.jpg"
		echo $cmd
		$cmd

	done

done

echo "DONE"
