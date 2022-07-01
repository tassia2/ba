#! /bin/bash
# basepath=
cd /home/tassia/Hiflow-2-22/ba-v2/build/examples/boussinesq2d
for folder in node_*; do 
    echo $folder; 
    mkdir /home/tassia/Hiflow-2-22/ba-v2/copy/$folder
    mkdir /home/tassia/Hiflow-2-22/ba-v2/copy/$folder/results_points
    for file in $folder/results_points/*; do
        cp $file /home/tassia/Hiflow-2-22/ba-v2/copy/$file;
    done;
done

