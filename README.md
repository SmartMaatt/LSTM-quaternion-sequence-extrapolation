
# Quaternion sequences in deep learning [QLSTM]

The repository includes implementations of quaternion networks and new QALE loss function, which calculates the error value based on the difference in angles between the result and the expected value. Procedures for performing the training and evaluation of predicting successive elements of a rotation sequence are also provided. The code was developed for a master's thesis: "**Quaternion sequences in deep learning**".


**Abstract**: The purpose of this paper is to verify the hypothesis that neural networks are capable of working with quaternion data. The main problem of this issue was the absence of common commercial solutions using artificial intelligence based on quaternion data. The decisive factor in this situation is the innovative nature of the solution, the popularity of which in scientific circles has only increased in the early 21st century. Research on systems integrating neural networks with quaternion data is important for the further development of the field of machine learning, making it possible to identify aspects in which the use of quaternions has an impact on the efficiency and precision of the network. For the purposes of the research, a model of the quaternion recurrent network QLSTM was developed, all of whose elements are in the form of quaternion data, and the key processes of machine learning were extended by the algebra of quaternions. A self-developed loss function was also implemented, determining the error based on the angle included between the resulting quaternion and the expected quaternion. The research was conducted on training sets that are quaternion sequences describing joint rotations over time. The experiments focused on comparing the results of networks equipped with Hamilton algebra, with basic recurrent networks in a regression problem. This problem involves predicting the further progress of rotation based on the input sequence of quaternions. The conclusions of the study define the advantage of the QLSTM network in the context of working with quaternion data, also highlighting the problems associated with it.

## Requirements
- Python 3.10
- Conda

## Instalation
- Open conda terminal in repository directory
- Type following command:

```
conda env create -f environment.yml
```
- Open souce code in IDE
- Select python interpreter: **qlstm**
- Enjoy ;)

## Credits
Thanks to Titouan Parcollet for your excellent PhD thesis and the code shared on github [[LINK]](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks). Without your help and knowledge this project would not exist. <3