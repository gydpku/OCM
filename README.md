# OCM
Here is the official implementation of the paper "Online Continual Learning thorough Mutual Information Maximization". This paper is accepted by ICML2022 as
a spotlight paper.
# Requirements
    pytorch<=1.6.0
    numpy==1.19.5
    scipy==1.4.1
    apex==0.1
# Usage
  To reproduce the results in the CIFAR10 setting (2 classes per task)
  
                python test_cifar10.py
                
  To reproduce the results in other setting (e.g. CIFAR100):
  
                python test_<dataset name>.py
                
  Note that the name of dataset is in lowercase. You can check them in the OCM file.

  
    
