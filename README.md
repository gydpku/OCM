# OCM
Here is the official implementation of the paper "Online Continual Learning thorough Mutual Information Maximization". This paper is accepted by ICML2022 as
a spotlight paper.
# Requirements
    pytorch<=1.6.0
    numpy==1.19.5
    scipy==1.4.1
    apex==0.1
    tensorboardX
    diffdist
# Usage
  To reproduce the results in the CIFAR10 setting (2 classes per task)
  
                python test_cifar10.py --buffer_size 1000
                
  To reproduce the results in other setting (e.g. CIFAR100):
  
                python test_<dataset name>.py --buffer_size xxx
                
  Note that the name of dataset is in lowercase. You can check them in the OCM file.
  
# Pseudo code for OCM (The simplest form to understand the method)
  Representation learning part
  
        x,y=x.cuda(),y.cuda() # get the new data input
        
        rotate_x,rotate_y=Rotation(x,y) # Using the rotation operation to create more pseudo classes
        
        hidden,hidden_aug= Basic_model(rotate_x,is_simclr=True), Basic_model(Aug(rotate_x),is_simclr=True) 
        #Aug is the data augmentation
        
        sim_matrix=torch.matmul(normalize(hidden),normalize(hidden_aug).t()) #similarity matrix
        
        InfoNce_loss_new = Supervised_NT_xent_uni(sim_matrix,labels=rotate_y,temperature=0.07) 
        # You can do the same thing for buffer data
        
  Forgetting loss part 
        
        mem_x,mem_y=mem_x.cuda(),mem_y.cuda() 
        # get the buffer data. You can choice the retrieval strategy by yourself.
        
        hidden_mem,hidden_mem_prev=Basic_model(mem_x,is_simclr=True), Previous_model(mem_x,is_simclr=True) 
        
        sim_matrix_prev= torch.matmul(normalize(hidden_mem),normalize(hidden_mem_prev).t())
        
        InfoNce_loss_prev = Supervised_NT_xent_pre(sim_matrix_prev, labels=mem_y, temperature=0.07)
        
        
        
                                                                     

  
    
