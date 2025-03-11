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
Jan Bentley was able to find a specific circuit in the convolutional layers of the impala model that activated strongly when the direction to go in was "up" and didn't activate when the cheese was in a similar spot but the agent would need to go in a different direction to get there. 
This indicates that a lot of the "planning" that the model would need to do was already handled in the CNN layers, since it would need to have already determined the location, and the fact that it would need to move in a direction that would allow it to move around the corner to get to the goal.

If this is the case, this indicates that it should be possible to train probes that can detect the exact direction the agent needs to move towards, and if it is possible to intervene with these probes, it could be possible to show fairly advanced control over the model.

However, something unique to the heist environment is the multiple goals that need to be reached one after another. One thing that would be a really great result is whether it is possible to get the model to pursue a different goal in the environment in a very clean way, rather than witsh the activation steering which is a very clumsy approach. If it's possible to show which goal its targeting from its weights and adjust this, this would be sufficient for me to go conclude the paper with this result. So that will be my initial target.

It would appear that this should be possible just using the CNN layers, if the layers have very strong directional features. Alternatively it's possible that the thing Jan found might not exist in the CNN layers of the heist but can only be found in the FC layers of the network. It's also possible that using the SAEs would surface these features.

Though it's also worth noting that the guys in the RL planning paper found that the monosemanticity of the neurons did not significantly increase between the SAE and the raw layers of the model, and that the raw layers were actually very interpretable. It's still tbd whether that is the case in the heist model as well. Confirming whether or not this is the case should be one of the first things that I do.

For now I won't be replicating the BImpala work, but I will also get Jon's feedback on this. It's possible that it's just a pareto improvement to work on BIMPALA since it has more interpretable qualities, and I can do all the same analyses in both cases.

Potentially it's worth training either way as it won't take too long to do.
# Interesting papers to follow up on:
[Understanding goal misgeneralisation in the procgen maze environment](https://www.alignmentforum.org/posts/vY9oE39tBupZLAyoC/localizing-goal-misgeneralization-in-a-maze-solving-policy)

[Bimpala work](https://arxiv.org/html/2412.00944#:~:text=We%20used%20the%20ProcGen%20environment,29)

### Paper notes Joseph Bloom Minigrid and the DT transformer:
In [this](https://www.lesswrong.com/posts/JvQWbrbPjuvw4eqxv/a-mechanistic-interpretability-analysis-of-a-gridworld-agent) paper, Joseph examines MemoryDT, which is a decision transformer trained on the minigrid memory task, where it is shown a specific object and then needs to go and select the correct object from a collection. It's an autoregressive trajectory modelling transformer. This is a very large architectural change from what is going on with the Impala model. 

In this case the model has to cross a gap where it can no longer see the original object and then select the correct object from a selection of options. It only obtains a reward when it chooses correctly. While crossing it needs to hold in its memory what the original object that it saw was.

This makes the model an interesting candidate for interpretability because one would presume the target goal must be held quite clearly in memory, and be identifiable as the model goes about its business. It is also a simple environment which has some significant upsides. Possibly most importantly the nature of the agent is closest to possible LLM style agents because the model is an agent simulator, that simply samples from trajectories that are likely to lead to success. This is similar to how a GPT LLM is a simulator of different characters that can be elicited through prompting or biased with fine-tuning, and what makes them capable general agents.

## Concrete next steps:
Result replication:
- [ ] Complete the feature visualisation of the CNN SAE layers. 
- [ ] Replicate whether it's possible to find these directional channels in the network.
	- [ ] This will involve setting up scenarios where the next key is around a little corner in different orientations and see if there is a specific layer that activates more significantly in each case.
- [ ] Replicate whether the SAEs are actually as monosemantic as the raw layers. 
	- [ ] This will involve just running the feature vis and seeing if the layers are significantly more interpretable in the SAE case. I will also need to double check the paper to see exactly how they reach this result.
- [ ] 