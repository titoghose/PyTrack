#!/bin/bash
fileid="1J0WL_OO4vO7Imq4lCFCreB9Nti3Kn5cy"
filename="tests/NTU_Experiment.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip ${filename} -d tests/