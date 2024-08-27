## Comprehensive Backdoor Framwork
This data poisoning-based backdoor attack framework target Spiking Neural Networks (SNNs), which covers three popular types of supervised learning rules (backporpagation-based, conversion-based, and hybrid learning methods), and two types of datasets (traditional image and neuromorphic datasets).
![image](https://github.com/bluefier/Comprehensive-Backdoor-Framework/blob/main/comprehensive%20backdoor%20framework.png)

* Step 1: **Data Poisoining**. Embed trigger patterns into clean data samples to construct poisoned data samples. Note that the samples in the datasets for SNN training is differ from that for ANNs, since they have an additional attribute, timestep.
* Step 2:**Backdoor Injection**. Training of models using malicious datasets containing poisoned data samples. Different learning rules have different training process.

## Code Structure

* LR_B   --backdoor injection for backpropagation-based learning rules
    * Traditional_image_datasets
    * Neuromotphic_datasets
* LR_C   --backdoor injection for conversion-based learning rules
* LR_H   --backdoor injection for hybrid learning rules

## Requirement

* python          3.9.18
* pytorch         2.0.1
* spikingjelly     0.0.0.0.14
