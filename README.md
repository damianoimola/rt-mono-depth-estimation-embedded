# MonoDeRT: A novel light-weight Real-Time architecture for Monocular Depth Estimation.

## Introduction
The topic of the project, initially was: *Real-time monocular depth estimation for low-resources embedded devices*, but due to **incompatibilities** between ARM 32-bit and the major libraries/framework to make inference, I had to slightly change the topic of the project.\
Now the title is: *MonoDeRT: A novel light-weight Real-Time architecture for Monocular Depth Estimation.*

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

![MonoDeRT architecture](https://raw.githubusercontent.com/damianoimola/rt-mono-depth-estimation-embedded/master/images/monodert_white_bg.jpg)


# How to run the code

## Model checkpoints
Every checkpoint is available in google drive at [this link](https://drive.google.com/drive/folders/1UmDH74_rk2Ef_6gE0_a_EHWhDzD02WN9?usp=sharing).

## Instructions
First of all, you need to convert the torch model to a ONNX model using the following command
```bash
python model_to_onnx.py [--ckpt <checkpoint_name>]
```
the checkpoint name, is the one without the extension `.ckpt`. If do not specify anything, the default loaded checkpoint is the last one: `mde60_kaggle.ckpt`.\
After executing this command, you'll obtain in the root folder a file named `model.onnx`, this will be the model used to make inference.

<div class="warning" style='background-color:#E9D8FD; color: black; border-left: solid #805AD5 4px; border-radius: 4px; padding:0.7em;'>
<span>
<p style='margin-top:1em; text-align:center'>
<b>Important note</b></p>
<p style='margin-left:1em;'>
    A file model.onnx is already available in the root, so if you do not need to load a specific checkpoint, you cna skip the first step.
</p></span>
</div>

> NOTE: in the root, it's still present an onnx model, so you can skip the first 

Now to execute the script to test the model with the camera, you need to use the following command
```bash
python cam_online_or_mean_onnx.py [--online <True/False> --fps_verbose <True/False>]
```
where `--online` is by default `True`, this helps in performance but maintaining a constant sampling rate that is defined a priori, otherwise we obtain a video that is sampled at right FPS, but with lower performances.\
While `--fps_verbose` works only for online mode abd it display FPS in command line. 