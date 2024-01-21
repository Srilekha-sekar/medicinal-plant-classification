# medicinal-plant-classification
#deeplearning, #cnn, #resnet50

to classify the medicinal I compared Resnet50 pretrained model and CNN model

Resnet50 pretrained model : 
ResNet (Residual Network) is a convolutional neural network that democratized the concepts of residual learning and skip connections. This enables to train much deeper models.

This is ResNet v1.5, which differs from the original model: in the bottleneck blocks which require downsampling, v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution. This difference makes ResNet50 v1.5 slightly more accurate (~0.5% top1) than v1, but comes with a small performance drawback (~5% imgs/sec) according to Nvidia.

What is a convolutional neural network (CNN)?
A convolutional neural network (CNN) is a category of machine learning model, namely a type of deep learning algorithm well suited to analyzing visual data. CNNs -- sometimes referred to as convnets -- use principles from linear algebra, particularly convolution operations, to extract features and identify patterns within images. Although CNNs are predominantly used to process images, they can also be adapted to work with audio and other signal data.

Dataset : https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/nnytj2v3n5-1.zip
