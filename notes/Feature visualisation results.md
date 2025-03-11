#masters #mech_interpretability 

![[Pasted image 20250210191957.png]]


This is before applying the sparse autoencoder, just using the base model as a sanity check

![[Pasted image 20250210192011.png]]


I was able to apply a similar technique to get some decentish results from imagenet, but I'm not sure to what extent the hyperpamaters need to change for the new model, or if there's other things I'm missing

at the very least we should expect to see like walls and game shapes and objects, but it's possible that if those don't appear that the network is actually too small to create human interpretable visualisations

If the feature visualisation angle continues to not work then it means we'd need to try and figure out if the SAE can tell us things in other ways such as potentially tracking specific objects in the environment

## claude summary

Let me summarize our approach and results in trying to visualize features in the Procgen Heist environment's CNN:

1. Initial Approach:

- Started with basic feature visualization using gradient ascent

- Used random initialization between 0-255

- Simple regularization with total variation and L2 losses

- Result: Got mostly white images, suggesting optimization was finding trivial solutions

2. Improvements Based on Distill Article:

- Added proper transformation robustness:

- Padding by 16 pixels to avoid edge artifacts

- Multiple scales (0.975, 1.025, 0.95, 1.05)

- Rotations (-5° to +5°)

- Two-pass jittering (16px then 8px)

- Adjusted optimization parameters:

- Increased steps to 2560

- Higher learning rate (0.05)

- Stronger regularization weights

- Current Results:

- Still not getting the rich, interpretable visualizations seen in the Distill article

- Getting mostly noise or simple patterns

- This suggests we're missing some crucial elements:

- Proper frequency space optimization (Fourier basis)

- Color decorrelation

3. Proper preconditioning of the optimization

- Key Differences from ImageNet Results:

- The Procgen environment is much simpler than ImageNet

- Our CNN architecture (IMPALA) is different from GoogLeNet used in the Distill article

- Our model is trained on a specific game environment rather than natural images

- The input resolution is lower (64x64 vs 224x224)

- Next Steps Could Include:

- Implementing proper Fourier space optimization

- Computing and applying color decorrelation from the training dataset

- Trying different layers and channels

- Experimenting with diversity terms to get multiple facets of each feature

- Collecting and analyzing actual game frames to better understand what patterns we should expect to see


![[Pasted image 20250211095914.png]]

![[Pasted image 20250211095855.png]]


![[Pasted image 20250211095839.png]]

![[Pasted image 20250211095819.png]]


![[Pasted image 20250211095745.png]]


![[Pasted image 20250211121624.png]]

![[Pasted image 20250211122653.png]]

![[Pasted image 20250211124105.png]]


## Gathering highly activating samples from the environment:

This is kind of interesting, I got a system working to extract patches from observations that are most strongly activating for different channels in layer 4 of the network. I first tried just finding the observations that were most strongly activating but since they all looked the same they weren't that helpful, but focusing in parts of the image that are most strongly activating is quite helpful.This is using a penalty term on the cosine similarity of samples we've chosen to encourage diversity
![[image (28).png]]

![[image (27).png]]

![[image (26).png]]

![[image (25).png]]

![[image (24).png]]![[Pasted image 20250212121853.png]]

![[Pasted image 20250212215730.png]]

![[Pasted image 20250212215658.png]]![[Pasted image 20250214102406.png]]