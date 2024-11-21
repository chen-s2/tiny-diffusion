### tiny diffusion

![Example](./sample.gif)

train a diffusion model from scratch in 2 hours on a basic gpu  


### train
1. download the [butterfly dataset](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification) from kaggle, or get your own 10k rgb images 
2. place all images (train and test set) in the same dir
3. pass the dir to train.py
4. run train.py, this is the training schedule I used:
   - 100 epochs with learning rate = 1e-4 
   - 50 epochs with learning rate = 0.5e-4
   - 50 epochs with learning rate = 0.2e-4 
5. for reaching the accuracy needed to generate samples like the above, my model reached a loss = 0.0289, and a clean loss = 0.2288.
### inference
1. run inference.py
2. see 'results' dir for the generated output images
