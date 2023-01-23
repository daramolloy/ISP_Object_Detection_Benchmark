# Evaluating the Impact of Varying ISP Parameters on Object Detection Performance


## Description

This project utilizes Fast-openISP (https://github.com/QiuJueqin/fast-openISP) and PyTorchs Torchvision and Hub to evaluate the impact that varying the parameters of image signal processing has on object detection performance. 
The work aims to characterise the deviation that exists between human vision and computer vision. 
To use, set a grid search of ISP parameters and provide raw images with accompanying annotations and this scripts provides thorough analysis of object detection performance.



## Dependencies

PyCOCOTools == 2.0.4

TIDE == https://github.com/dbolya/tide/pull/21

pytorch == 1.11.0

torchvision == 0.12.0

tqdm == 4.64.0

Just to note that your TIDE library will need to be updated as suggested below due to the repo no longer being updated.

https://github.com/dbolya/tide/pull/21


## Authors

Contributors names and contact info

Dara Molloy (https://www.linkedin.com/in/daramolloy/)

Brian Deegan (https://www.linkedin.com/in/brian-deegan-54928b60/)


## Acknowledgements

Massive thanks to Qiu Jueqin for creating fast-openISP (https://github.com/QiuJueqin/fast-openISP). 
Some changes were made to the original library, primarily making the pipeline threads run asynchronously as this gave a big time improvement as well as fixing edge enhancement and adding openCVs CFA interp methods.