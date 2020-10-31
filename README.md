# Echolocation-AEC
This is the project folder for the journal articel "Active head rolls enhance sonar-based auditory localization performance"
## Table of contents
* [Introduction](#introduction)
* [Description](#description)
* [Setup](#setup)
* [Software](#software)
* [Acknowledgements](#acknowledgements)

## Introduction
This project provide data and code related to the journal pulblication. The folder "Figure_data" provides the data for all the figures shown in the article. The provided codes form the Active Efficient Coding (AEC) model.
	
## Description
Here, we will describe the main codes required for training and testing the proposed model.
| Code                        | Description                    |
| -------------               |:------------------------------:|
| Main.m                      | Main file required for training|
| Generate_test_trials.m      | Generate testing trials        |
| Model.m                     | Model class                    |
| ASSOMOnline.m               | GASSOM                         |
| CActorG.m                   | Actor (Reinforcement Learning) |
| CCritic.m                   | Critic(Reinforcement Learning) |
	
## Setup
To run this project, download the project folder and run:

```
$ Main
$ Generate_test_trials
```
The program "Main.m" will save the learned model to the folder "Subject_{subject number}_trail_{trial number}_sigma_{roll angle standard deviation}". "Generate_test_trials.m" will generate the tresting trial trajectories and plot the trajectories. 

## Software
All the codes are tested with the MATLAB version R2017a (64-bit) under ubuntu 16.04 LTS.

## Acknowledgements
We would like to acknowledge the graduate students contributed to the Active Efficient Coding (AEC) model over the years under the supervision of Prof. Jochen Triesch and Prof. Bertram E. Shi.

* Yu Zhao
* Thusitha N. Chandrapala
* Chong Zhang‬
* Qingpeng Zhu
* Céline Teulière
* Luca Lonini

