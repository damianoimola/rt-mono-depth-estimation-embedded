# Real-time monocular depth estimation for low-resources embedded devices


The project aims to design a model architecture for Real-Time Monocular Depth Estimation for low-resources embedded devices (i.e. Raspberry Pi Zero: 512 MB of RAM and 1GHz, single-core CPU).
The task is hard due to the unusual resources constraints, so to pursue it I'll use optimization techniques learned during lectures and through some recent papers that explores the paradigm ranging from Micro-Controller Units (MCUs) to Embedded devices.
Moreover there are some architectures that tries to decrease latency by cropping some output layers at inference, this is really helpful in order to achieve the real-time performances (i.e. >= 30 FPS) on such devices.
For the data I'm going to use, I'll try various datasets: DIODE, NYU Depth and KITTI Depth.
