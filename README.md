# Grasp-lift-replace_EEG_Detection
## Overview
The project is paart of my work as an AI/ML Intern at [Reflections Info Systems PVT LTD](https://reflectionsglobal.com/).
The problem is a  competition is sponsored by the WAY Consortium (Wearable interfaces for hAnd function recoverY; FP7-ICT-288551), which was listed in Kaggle along with the dataset. [Grasp-and-lift EEG_Detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection)

## Problem Statement
To identify when a hand is grasping, lifting and replacing an object using EEG data mainly to predict or identify the following events
- HandStart
- FirstDigitTouch
- BothStartLoadPhase
- LiftOff
- Replace
- BothReleased
## EEG Dataset
-Each row is a time frame ( sampling rate = 500 Hz) i.e that means 500 time frame of 0.002 seconds are recorded each second.
-Columns: 32 channels(Electrodes) + 6 Labels(Phases of the movement)
-There are 12 subjects(Individuals) with 10 series per subject. Among these 10, 1-8 is used for training and 9-10 for testing.
Signals

Labels
![ss1](https://user-images.githubusercontent.com/84126934/145026798-8dc7eb32-0211-4a3b-8afc-2b6d6166a1cf.png)

## References
- [Kaggle Meetup: Grasp-and-List](https://www.youtube.com/watch?v=hjJ4eJ72aUQ&t=3247s)
- [Grasp-and-lift EEG_Detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection)
