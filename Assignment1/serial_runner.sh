#!/bin/bash

echo -n "Please enter a config folder: "
read folder
slash="/"

for file in $folder/*;
do

    filename="$(echo $file | awk -F/ '{print $NF}')"
    echo $filename
    filename=$(echo $filename | cut -d'.' -f 1)
    echo $filename
    python main.py $file > "${folder}/output_${filename}.out"
done

read -p "Press any key to continue"