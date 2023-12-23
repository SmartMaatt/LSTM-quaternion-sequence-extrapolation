<h1 align="center">LSTM Quaternion Sequence Extrapolation (QLSTM)</h1>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#screenshots">Screenshots</a> •
  <a href="#installation">Installation</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#acknowledgements">Acknowledgements</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  <img src="https://img.shields.io/badge/Author-SmartMatt-blue" />
</p>

## Overview
This repository is dedicated to the exploration and implementation of LSTM networks using quaternion data, as part of the research project titled "**Quaternion sequences in deep learning**". The primary focus is on the application of quaternion algebra in deep learning for predicting sequences, particularly in the context of rotation sequences.

### Abstract
The purpose of this paper is to verify the hypothesis that neural networks are capable of working with quaternion data. The main problem of this issue was the absence of common commercial solutions using artificial intelligence based on quaternion data. The decisive factor in this situation is the innovative nature of the solution, the popularity of which in scientific circles has only increased in the early 21st century. Research on systems integrating neural networks with quaternion data is important for the further development of the field of machine learning, making it possible to identify aspects in which the use of quaternions has an impact on the efficiency and precision of the network. For the purposes of the research, a model of the quaternion recurrent network QLSTM was developed, all of whose elements are in the form of quaternion data, and the key processes of machine learning were extended by the algebra of quaternions. A self-developed loss function was also implemented, determining the error based on the angle included between the resulting quaternion and the expected quaternion. The research was conducted on training sets that are quaternion sequences describing joint rotations over time. The experiments focused on comparing the results of networks equipped with Hamilton algebra, with basic recurrent networks in a regression problem. This problem involves predicting the further progress of rotation based on the input sequence of quaternions. The conclusions of the study define the advantage of the QLSTM network in the context of working with quaternion data, also highlighting the problems associated with it.

### Key Features
- **Quaternion Networks**: Implementation of LSTM networks using quaternion data.
- **QALE Loss Function**: A custom loss function for evaluating prediction accuracy based on quaternion angles.
- **Quaternion Data Handling**: Techniques for processing and manipulating quaternion data within neural networks.

## Screenshots
![Extrapolated skeleton](https://smartmatt.pl/github/qlstm/extrapolated-skeleton.png)
*Animation presentation of an extrapolated skeleton from a motion capture recording in Mokka software.*

## Installation
1. **Set Up Conda Environment**:
   - Open a Conda terminal in the repository directory.
   - Create the environment using the provided `environment.yml` file:
     
     ```bash
     conda env create -f environment.yml
     ```
2. **IDE Configuration**:
   - Open the source code in your preferred IDE.
   - Set the Python interpreter to `qlstm`.

## Usage
- The repository includes scripts and modules for training and evaluating the QLSTM model.
- Detailed usage instructions are provided in the `thesis.pdf` file.

## Requirements
- Python 3.10
- Conda

## Acknowledgements
Special thanks to Titouan Parcollet for the foundational work in quaternion neural networks. His PhD thesis and the code available at [Orkis-Research/Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks) were invaluable resources in this project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
&copy; 2023 Mateusz Płonka (SmartMatt). All rights reserved.
<a href="https://smartmatt.pl/">
    <img src="https://smartmatt.pl/github/smartmatt-logo.png" title="SmartMatt Logo" align="right" width="60" />
</a>

<p align="left">
  <a href="https://smartmatt.pl/">Portfolio</a> •
  <a href="https://github.com/SmartMaatt">GitHub</a> •
  <a href="https://www.linkedin.com/in/mateusz-p%C5%82onka-328a48214/">LinkedIn</a> •
  <a href="https://www.youtube.com/user/SmartHDesigner">YouTube</a> •
  <a href="https://www.tiktok.com/@smartmaatt">TikTok</a>
</p>

