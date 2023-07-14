<?php

require 'vendor/autoload.php';

use Keras\Keras;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use OpenCV\Capture\Capture;
use OpenCV\Classifier\CascadeClassifier;
use OpenCV\Matrix;

$emotion_dict = [0 => "Angry", 1 => "Disgusted", 2 => "Fearful", 3 => "Happy", 4 => "Neutral", 5 => "Sad", 6 => "Surprised"];

// Load the JSON model
$json_file = fopen('emotion_model_2.json', 'r');
$loaded_model_json = fread($json_file, filesize('emotion_model_2.json'));
fclose($json_file);
$emotion_model = new Keras($loaded_model_json);

// Load the weights into the model
$emotion_model->loadWeights("emotion_model_2.h5");
echo "Loaded model from disk\n";

// Start the webcam feed
$cap = new Capture(0);
// Or pass video path
// $cap = new Capture("Mtumba_clothes_abroad.mp4");

while (true) {
    // Find Haar cascade to draw bounding box around face
    $frame = $cap->queryFrame();
    $gray_frame = $frame->cvtColor(\OpenCV\ColorConversionCodes::COLOR_BGR2GRAY);

    $face_detector = new CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml');

    // Detect faces available on camera
    $faces = $face_detector->detectMultiScale($gray_frame, 1.3, 5);

    // Preprocess faces
    foreach ($faces as $face) {
        $x = $face->x;
        $y = $face->y;
        $w = $face->width;
        $h = $face->height;
        $frame->rectangle($face->x, $face->y - 50, $face->x + $face->width, $face->y + $face->height + 10, new \OpenCV\Scalar(0, 255, 0), 4);
        $roi_gray_frame = $gray_frame->getImageROI(new \OpenCV\Rect($x, $y, $w, $h));
        $cropped_img = $roi_gray_frame->resize(new \OpenCV\Size(48, 48))->expandDims(0)->expandDims(-1);

        // Predict emotion
        $emotion_prediction = $emotion_model->predict($cropped_img);
        $maxindex = array_keys($emotion_prediction[0], max($emotion_prediction[0]))[0];
        $text = $emotion_dict[$maxindex] . " " . print_r($emotion_model->predict($cropped_img), true);
        $frame->putText($text, new \OpenCV\Point($x + 5, $y - 20), new \OpenCV\Font(), 1, new \OpenCV\Scalar(255, 0, 0), 2, \OpenCV\LINE_AA);
    }

    $frame->showImage("Emotion Detection");
    $key = cvWaitKey(1);
    if ($key == 'q') {
        break;
    }
}

$cap->release();
cvDestroyAllWindows();

?>