mkdir -p data
cd data

wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
wget -c http://pjreddie.com/media/files/VOC2012test.tar
tar -xvf VOC2012test.tar

wget -c wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
tar -zxvf benchmark.tgz
mkdir -p VOC_AUG VOC_AUG/images VOC_AUG/labels
cp -riv benchmark_RELEASE/dataset/*txt VOC_AUG/
cp -riv benchmark_RELEASE/dataset/img/* VOC_AUG/images/
cd ..