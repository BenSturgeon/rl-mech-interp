#masters #mech_interpretability #reinforcement_learning 

Working documents:
[[Feature visualisation results]]


## Project description
Currently I am working on the procgen heist environment and trying to develop methods for applying mechanistic interpretability techniques to it. 

Currently I am using an Impala model and trying to interpret how it identifies which entity in the environment is the current goal.

I am applying Sparse Autoencoders to the CNN layers of the model, and am considering trying to find ways to extract information from the FC layers of the model as well. 

ImpalaCNN architecture:

CustomCNN:
- Input: Image (H,W,C) → Normalization [0,1] → Format adaptation
- Conv Block 1: Conv(C→16, 7×7) → ReLU → LPPool(2×2, s=2)
- Conv Block 2: Conv(16→32, 5×5) → ReLU → Conv(32→32, 5×5) → ReLU → LPPool(2×2, s=2)
- Conv Block 3: Conv(32→32, 5×5) → ReLU → LPPool(2×2, s=2) 
- Conv Block 4: Conv(32→32, 5×5) → ReLU → LPPool(2×2, s=2)
- Flatten → Linear(flattened→256) → ReLU
- Linear(256→512) → ReLU
- Dual heads: Policy(512→num_outputs) & Value(512→1)



Currently I am confused about what a good question to ask is. Like maybe just figuring out goal representation is valuable enough as a goal, but I'm not that sure. I think it'd also be really interesting to be able to interpret the specific mechanisms that the model makes use of in reaching its goals. Such as whether it measures say Euclidean distance between specific objectives or whether there's a mechanism it uses to detect whether there's a block between it and a certain goal. And I'm not too sure what the right direction to take is. 

The last things I've been working on have involved training a sparse autoencoder on the CNN layers and doing feature visualisation on the CNN layers in order to try and determine what they're actually doing, and if the SAE trained on that layer and then exploded out yields more interesting and interpretable results in terms of the CNN layers.

I am also curious about whether a bilinear MLP approach can lead to some interesting results that might help interpret the model much more directly.



## Literature review + Deep Research
[Deep Research analysis](https://chatgpt.com/share/67cf1d08-a214-800d-9de7-87058edf9f61)

The overall analysis here provides a surprisingly interesting landscape of existing work in RL highlighting a few papers I wasn't aware of, particularly the goal misgeneralisation work by Jan Betley which I think is very relevant and has some great ideas for my research.

The biggest thing that I'm most excited about is the potential of combining techniques from bilinear MLPs and Sparse Autoencoders. It does certainly seem as though techniques from bilinear networks can be leveraged to give highly interpretable results though the underlying math that goes into the process is quite tough to parse.

There are also a lot of interesting questions for me regarding the Sokoban environment where the authors were able to in fact replace certain parts of the actual network with python code and preserve a good deal of the functionality which demonstrates a high degree of network comprehension. 

In this case the strongest techniques they used were linear probes that were applied to create a specific intervention which could then be analysed. This could yield some useful insights as I'm not sure if the exact same results could be leveraged, but certainly i think some creative probe experiments could lead to some really interesting results in and of themselves.
## Highlights from Deep Research

**Combine Top-Down and Bottom-Up Approaches**: A productive strategy is to use _concept probes_ (top-down) in tandem with _mechanistic decomposition_ (bottom-up). For example, first identify a concept of interest (like “goal location” or “has key”) and train a simple probe to find where it’s encoded in the network. Then, in that layer, apply methods like SVD or sparse coding to break down the activations or weights tied to the concept. This was the protocol followed in the bilinear interpretability work – they probed for cheese location, then used the probe’s weights to guide an eigen decomposition of the preceding layer ​[arxiv.org](https://arxiv.org/html/2412.00944#:~:text=4)
[arxiv.org](https://arxiv.org/html/2412.00944#:~:text=Perform%20an%20eigendecomposition%20towards%20the,that%20write%20to%20the%20probe), yielding human-interpretable filters. Such an approach ensures you **find meaningful directions** in the network rather than arbitrary ones.

## Possible interesting directions to go in:
Training probes to track the position of the next entity
Training probes to track the positions of the non-next entities in the environment


## Key working insights
Jan Bentley was able to find a specific circuit in the convolutional layers of the impala model that 


## Interesting papers to follow up on:
[Understanding goal misgeneralisation in the procgen maze environment](https://www.alignmentforum.org/posts/vY9oE39tBupZLAyoC/localizing-goal-misgeneralization-in-a-maze-solving-policy)

[Bimpala work](https://arxiv.org/html/2412.00944#:~:text=We%20used%20the%20ProcGen%20environment,29)

