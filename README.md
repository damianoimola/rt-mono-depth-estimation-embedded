# MonoDeRT: A novel light-weight Real-Time architecture for Monocular Depth Estimation.

## Introduction
The topic of the project, initially was: *Real-time monocular depth estimation for low-resources embedded devices*, but due to incompatibilities between RaspiOS and the major libraries/framework to make inference, I had to slightly change the topic of the project.\\
Now the title is: *MonoDeRT: A novel light-weight Real-Time architecture for Monocular Depth Estimation.*.

### Old abstract
The project aims to design a model architecture for Real-Time Monocular Depth Estimation for low-resources embedded devices (i.e. Raspberry Pi Zero: 512 MB of RAM and 1GHz, single-core CPU).
The task is hard due to the unusual resources constraints, so to pursue it I'll use optimization techniques learned during lectures and through some recent papers that explores the paradigm ranging from Micro-Controller Units (MCUs) to Embedded devices.
Moreover there are some architectures that tries to decrease latency by cropping some output layers at inference, this is really helpful in order to achieve the real-time performances (i.e. >= 30 FPS) on such devices.
For the data I'm going to use, I'll try various datasets: DIODE, NYU Depth and KITTI Depth.

### New abstract
The project aims to design a model architecture for Real-Time Monocular Depth Estimation that is lightweight. In particulare I'll leverage the smoothness of the bicubic interpolation in order to make accurate, fast and smooth predictions.
This is an ill-posed task that is hard due to the lightweight and fast inference constraints, so to pursue it I'll use optimization techniques learned during lectures and through some recent papers that explores the paradigm ranging from Micro-Controller Units (MCUs) to Embedded devices.
Moreover there are some architectures that tries to decrease latency by cropping some output layers at inference, this is really helpful in order to achieve the real-time performances (i.e. >= 30 FPS) on such devices.
For the data I'm going to use, NYU Depth V2 (a subsample of 50.000 samples against 408.000).


# Proposed method
The novel model called MonoDeRT (that stands for: Monocular Depth Estimation Real-Time) is a pyramidal encoder-decoder model with residual connections, that leverage the bicubic interpolation in both upsampling and downsampling. Follows its graphical representation

![MonoDeRT architecture](https://raw.githubusercontent.com/damianoimola/rt-mono-depth-estimation-embedded/master/images/monodert.drawio.svg)
