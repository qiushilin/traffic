**RECOGNISING TRAFFIC SIGNS**<br>
<b>Dataset: <a href="https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign">German Traffic Sign Recognition Benchmark (GTSRB)</a></b><br><br>

The German Traffic Sign Benchmark is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. 

<li>43 classes</li>
<li>More than 50,000 images in total</li>


<br/>
Accuracy achieved: 97.5%
<br/>

Usage: python traffic.py data_directory [model.h5]<br>
<li>data_directory is the directory where the photos are stored</li>
<li>model.h5 (optional) is the filename in which user wants to store the final model</li>
e.g. python3 traffic.py gtsrb model.h5
