# Chào các bạn,
# Hôm nay mình sẽ giới thiệu cho các bạn về chủ đề nhận dạng cảm xúc qua khuôn mặt, sử dụng 2 phương pháp chính là SVM và CNN.
# Mục tiêu của bài viết:
##  + So sánh phương pháp SVM và CNN trong nhận dạng cảm xúc qua khuôn mặt.
##  + So sánh phương pháp CNN cơ bản và CNN cơ bản kết hợp các đặc trưng truyền thống.

Để làm rõ 2 mục tiêu trên, mình sẽ giới thiệu mô hình kiến trúc mình triển khai (Mô hình này mình đã tìm hiểu và tham khảo bài báo: [[Facial Expression Recognition using Convolutional Neural Networks: State of the Art, Pramerdorfer & al. 2016]](https://arxiv.org/abs/1612.02903) - Đây là bài báo uy tin, được đăng lên tạp chí nổi tiếng và tới thời điểm hiện tại có 71 bài báo khác tham chiếu + tham khảo đến nó)

Mô hình CNN kết hợp đặc trưng truyền thống cho hệ thống nhận dạng cảm xúc qua khuôn mặt
![Model's architecture](CNN_models_architecture.png)

Mô hình SVM thiết kế cho hệ thống nhận dạng cảm xúc qua khuôn mặt
![Model's architecture](svm.png)

Mô hình CNN cơ bản thiết kế cho hệ thống nhận dạng cảm xúc qua khuôn măt
![Model's architecture](cnn_coban.png)

Sau khi thực hiện triển khai mô hình, và thực hiện cài đặt, ta được kết quả dưới đây :
![Model's architecture](ketqua_fer.png)

# -> Nhìn vào bảng kết qủa ta đã chứng tỏ được kết quả mục tiêu thực hiện.


## <a name="install">4.1. Install dependencies</a>

- Tensorflow
- Tflearn
- Numpy
- Argparse
- [optional] Hyperopt + pymongo + networkx
- [optional] dlib, imutils, opencv 3
- [optional] scipy, pandas, skimage

Better to use anaconda environemnt to easily install the dependencies (especially opencv and dlib)

## <a name="data">4.2. Download and prepare the data</a>

1. Download Fer2013 dataset and the Face Landmarks model

    - [Kaggle Fer2013 challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
    - [Dlib Shape Predictor model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

2. Unzip the downloaded files

    And put the files `fer2013.csv` and `shape_predictor_68_face_landmarks.dat` in the root folder of this package.

3. Convert the dataset to extract Face Landmarks and HOG Features
    ```
    python convert_fer2013_to_images_and_landmarks.py
    ```
    
    You can also use these optional arguments according to your needs:
    - `-j`, `--jpg` (yes|no): **save images as .jpg files (default=no)**
    - `-l`, `--landmarks` *(yes|no)*: **extract Dlib Face landmarks (default=yes)**
    - `-ho`, `--hog` (yes|no): **extract HOG features (default=yes)**
    - `-hw`, `--hog_windows` (yes|no): **extract HOG features using a sliding window (default=yes)**
    - `-hi`, `--hog_images` (yes|no): **extract HOG images (default=no)**
    - `-o`, `--onehot` (yes|no): **one hot encoding (default=yes)**
    - `-e`, `--expressions` (list of numbers): **choose the faciale expression you want to use: *0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral* (default=0,1,2,3,4,5,6)**

    Examples:
    ```
    python convert_fer2013_to_images_and_landmarks.py
    python convert_fer2013_to_images_and_landmarks.py --landmarks=yes --hog=no --how_windows=no --jpg=no --expressions=1,4,6
    ```
    The script will create a folder with the data prepared and saved as numpy arrays.
    Make sure the `--onehot` argument set to `yes` (default value)

## <a name="train">4.3. Train the model</a>
1. Choose your parameters in 'parameters.py'

2. Launch training:

```
python train.py --train=yes
```

The variable `output_size` in parameters.py (line 20), should correspond to the number of facial expressions you want to train on. By default it is set to 7 expressions.

3. Train and evaluate:

```
python train.py --train=yes --evaluate=yes
```

N.B: make sure the parameter "save_model" (in parameters.py) is set to True if you want to train and evaluate

## <a name="optimize">4.4. Optimize training hyperparameters</a>
1. For this section, you'll need to install first these optional dependencies:
```
pip install hyperopt, pymongo, networkx
```

2. Lunch the hyperparamets search:
```
python optimize_hyperparams.py --max_evals=20
```

3. You should then retrain your model with the best parameters

N.B: the accuracies displayed are for validation_set only (not test_set)

## <a name="evaluate">4.5. Evaluate a trained model (calculating test accuracy)</a>

1. Modify 'parameters.py':
 
Set "save_model_path" parameter to the path of your pretrained file.

2. Launch evaluation on test_set:

```
python train.py --evaluate=yes
```

## <a name="recognize-image">4.6. Recognizing facial expressions from an image file</a>

1. For this section you will need to install `dlib` and `opencv 3` dependencies

2. Modify 'parameters.py':

Set "save_model_path" parameter to the path of your pretrained file

3. Predict emotions from a file

```
python predict.py --image path/to/image.jpg
```

## <a name="recognize-video">4.7. Recognizing facial expressions in real time from video</a>

1. For this section you will need to install `dlib`, `imutils` and `opencv 3` dependencies

2. Modify 'parameters.py':

Set "save_model_path" parameter to the path of your pretrained file

3. Predict emotions from a file

```
python predict-from-video.py
```
A window will appear with a box around the face and the predicted expression.
Press 'q' key to stop.

N.B: If you changed the number of expressions while training the model (default 7 expressions), please update the emotions array in `parameters.py` line 51.


# <a name="contrib">5. Contributing</a>

Some ideas for interessted contributors:
- Automatically downloading the data
- Adding data augmentation?
- Adding other features extraction techniques?
- Improving the models

Feel free to add or suggest more ideas.
Please report any bug in the [issues section](https://github.com/amineHorseman/facial-expression-recognition-using-cnn/issues).
