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

However, something unique to the heist environment is the multiple goals that need to be reached one after another. One thing that would be a really great result is whether it is possible to get the model to pursue a different goal in the environment in a very clean way, rather than with the activation steering which is a very clumsy approach. If it's possible to show which goal its targeting from its weights and adjust this, this would be sufficient for me to go conclude the paper with this result. So that will be my initial target.

It would appear that this should be possible just using the CNN layers, if the layers have very strong directional features. Alternatively it's possible that the thing Jan found might not exist in the CNN layers of the heist but can only be found in the FC layers of the network. It's also possible that using the SAEs would surface these features.

Though it's also worth noting that the guys in the RL planning paper found that the monosemanticity of the neurons did not significantly increase between the SAE and the raw layers of the model, and that the raw layers were actually very interpretable. It's still tbd whether that is the case in the heist model as well. Confirming whether or not this is the case should be one of the first things that I do.

For now I won't be replicating the BImpala work, but I will also get Jon's feedback on this. It's possible that it's just a pareto improvement to work on BIMPALA since it has more interpretable qualities, and I can do all the same analyses in both cases.

Potentially it's worth training either way as it won't take too long to do.

## SAE Feature visualisation experiment results
One of the best ways to determine what an SAE has learned is to try and visualise the features it has learned directly. During training the SAE is encouraged to develop channels that uniquely activate in a particular circumstance by penalising the total value of all aggregated activations in the encoder. This means that a particular signal is best sent by having as many layers produce no signal at all as possible, while a few channels in combination will activate at a given timestep, meaning they have to specialise in their function. 

Continued ability to perform the original function of the network is typically maintained by expanding the total number of channels. While often a given channel may serve multiple functions due to the number of features that need to be captured being greater than the number of channels, expanding the number of channels allows a given feature to be captured and exclusively represented in a single channel.

By visualising each neuron we can see what it has potentially learned. One of the most illuminating aspects of this is to expose dead neurons on the network, which we see a significant number of in the network (roughly 25%). This is typical in SAEs where for some reason some channels do not learn any feature. This can be because there are sufficient features encoded that the loss of reconstructing the network was met without needing to encode all of the original features, and, perhaps some of the less meaningful features could simply left out, or because there are in fact fewer learned features than there were channels in the network.

We do in fact see quite striking visualisations from the SAE, though further work on tuning the colours would yield better results. These results are also significantly clearer in the SAE than they were in the original model.

[[SAE feature visualisation results]]

% Images aren't currently rendering and won't be uploaded for now

I will refrain from speculating on the meaning of particular features currently, as it is difficult to determine exactly what each channel represents currently. 

this investigation will require matching with samples from the environment like we see here.
![[image (25).png]]

Possible experiments that would be useful to run here would be trying to do feature visualisation on specific patches of images from the environment as well, given that this was so successful when sampling from the dataset and yielded such strong activations as well. If we could color correct from the environment correctly this would yield even stronger results.

It would also be helpful to do the same with actual frames from the environment and see when each of the channels from the environment activates the most strongly and compare this to the base model. If we could gather large enough sets of samples this could yield very useful patterns. A similar approach could likely be used on the MLP layers as well.



## Next steps
I have now pushed the feature visualisation stuff pretty near to its limits. Now I think the right direction is to try and explore how different sets of SAE features work together. In particular, trying to capture if there is an SAE feature that correlates with the agent needing to move up, down, down, left and right, and also if we can identify a specific channel for each of the different entities and potentially switch between them.

## A more complete write up of the visualisation results
Feature visualisation is an extremely useful tool in mechanistic interpretability due to it giving you some direct insight into what a particular part of the network is doing. It can be difficult however, due to implementation challenges, as well as misleading in terms of what is actually happening. It can be useful as an initial indicator for signs of life when doing mechanistic interpretability work. 

## Feature visualisation methodology
There are two main approaches to feature visualisation, synthetic visualisation and finding maximally activating samples. Each has different strengths and weaknesses. Synthetic visualisation can yield more precise reflections of exactly what a channel is visualising, while max activating samples can be much more clearly interpretable.

Synthetic visualisation involves using gradient descent to iteratively adjust an input image with the optimisation target of getting the strongest possible average activation across a channel in a target channel, layer our single neuron. In our investigation we focused only on whole channels. This means we would sum the strength of activations across the width and height of a given channel and then multiply the mean activation by -1 as our loss, which we would then try and minimise.

Max activation sampling involves selecting samples from a dataset that produce the highest mean activation score across a given layer. To avoid duplicates we apply a diversity score which works as a penalty against selecting examples that are very similar to existing samples we've selected using a cosine similarity score. This diversity score can be applied as a variable weighting to indicate the importance of diversity.

We apply a variation of max activation sampling where we  samples patches from an image to identify specific features that strongly activate specific parts of a channel. To do this we pass inputs from the environment through the model and then identify when a specific part neuron in a channel is very strongly activating. We then assess where there is a patch in the channel that has an unusually high average activation and then select this area within the original image to visualise the region. We then rescale the patches so they are all the same size. The same diversity score as above is also applied.

Our synthetic visualisation is enhanced by following state of the art techniques to improve the fidelity of the results such as applying jitter, rotations, and other transformations, and by applying a process to decorrelate the visualisations from each other in the colour spectrum. This is because when trying to develop maximally activating inputs there is significant overlap between when a certain colour is activating a part of the image and part of the strength of the activation is from a correlation with other nearby colours in the rgb spectrum. By first decorrelating our image, essentially whitening the image we are able to individually maximise for each colour, and then recolour the images according to what the actual colours would be in the environment which gives us a more true to life visualisation.


## Feature visualisation results
We find that the SAE does provide significantly more clear visualisations than in the original model. where a great deal of the visualisations primarily consist of noise.

Compare the fidelity of features between the channels in the SAE with the raw channels in the network. There is clearly some specialisation in the SAE channels, and the presence of dead neurons is also indicative that our SAE is working as expected, as often there is simply not sufficient need for a particular feature to be encoded to embed itself into a feature. Sometimes the cost of integrating into another channel is sufficient for it to accept the sparsity penalty, while in other cases the feature itself may simply not be important enough. Dead neurons in SAEs are very common and thus are some confirmation that our SAE is working as expected.



### Applying decorrelation to enhance visual features

I found that applying the color decorrelation significantly improved the quality of the results, though their actual interpretability is still fairly questionable. This is definitely superior to what I was getting before so any future feature visualisation will make use of this.

### Experiments in patch visualisation for the SAE.
This was very disappointing as the results were fairly uniform across features in the SAE. I found that there was also little difference between the decorrelated results and the standard results.

### Finding max activating patches in the SAE 
This worked pretty great, which was not surprising. I am not sure if it's much better than in the standard cases but the results are fairly clear at least. I think this is probably the most promising direction at this point.


## Activation manipulation
One key experiment was to try and achieve control of the network by modifying a single channel in the network as they were able to do in the original maze solving network paper by Turntrout et al.

To work towards achieving this, it was necessary to identify channels that seemed to track a destination that the model was heading towards. In some sense it seems very necessary that the model be able do identify far off destinations to move towards in order to solve the maze, and it would need to do this from an array of different entities that it would need to target at different times (the keys).

The simplest method I could think of for doing this was to simply move different target entities around while looking at the activations and seeing how they might change. To achieve this we go in and modify the environment directly, and then create a sequence of mazes with the object of interest moving in a clockwise pattern around the maze. We can then just see which channels seem to most closely track different entities. Of particular interest were channels that tracked multiple entities closely. 

Channel 95 seemed to do this quite closely. It would track each key as it was moved around the maze. However, these results can be difficult to trust as they vary depending on the position of the player relative to the object. 

This is in contrast to the results of Turntrout et al where they found that the model had a very specific channel that would very consistently track the position of the cheese. This model is significantly less transparent in this regard as we see that patterns tracking specific entities seem to change depending on a range of factors such as the player being in a certain location on the map. 

This was the key motivation to train the sparse autoencoder, in the hope that it would concentrate the role of tracking entities into more clearly identifiable channels.

### Identifying channels to modify
To identify which channels to modify we create a simple experiment where we move the object of interest around a square to and pass the observations generated by these mazes into the model. We then see which channels seem to most closely track the object by seeing how predictive the strength of activations in a channel are with the objects position. We used a few metrics to do this, specifically equivariance and spatial overlap scores.

Equivariance is calculated by tracking the position of the centroid of the entity and the centroid of the mass of the activations. We determine the movement vectors of each of these as the object changes position and calculate the cosine similarity between the two vectors to determine the extent to which transformations in activation space translate to transformations of the object. 

This gives us a good sense of when a channel is "tracking" the object.

### Performing spatial interventions on the model

To test whether we could find a specific channel that would closely track the current objective, we created a new test that allowed us to intervene on a specific channel at a specific point to see if we could alter the behaviour predictably. To do this we would specify a single spot on the channel that would have a very high value, and mask out all other values to 0. We'd then run the episode with this static change while allowing it to pursue some other objective. The area of activation would be some area other than where the target object was. 

We created the experiment such that multiple channels could be modified at once. The choice of channels that would be altered was based on those channels that had scored highest on our tracking metrics.

Initially these results did not yield any effect on the trajectory of the agent through the maze at all. This seemed in stark contrast to the results attained by Turntrout et al with the single target maze environment. 

This could mean a number of things: 
* The model navigates through a different mechanism than the cheese finding model.
* There are sufficient redundancies in other channels that the modification of that one does not create a noticeable difference.
* The channels that are responsible for navigating do not rely on the parts of the network that specifically track certain entities.
* The fact that there are fewer channels in this network compared to the network they were using may mean that compressing more features into fewer channels makes it impossible to cleanly separate it out.

When we originally made the decision to use a smaller, more compressed and more interpretable model this may have made the task harder. This highlights a fundamental tension in this kind of interpretability work. Having single clear representations is made easier in some cases by expanding the number of neurons, but is also in some senses made easier to manipulate by forcing specific functionality into a smaller number of neurons.

## Next steps from the spatial intervention results

At this point the best action to take seems to be training the decision transformer and training up some probes and trying to do the interventions that way. This seems more exciting than the current direction.





# Interesting papers to follow up on:
[Understanding goal misgeneralisation in the procgen maze environment](https://www.alignmentforum.org/posts/vY9oE39tBupZLAyoC/localizing-goal-misgeneralization-in-a-maze-solving-policy)

[Bimpala work](https://arxiv.org/html/2412.00944#:~:text=We%20used%20the%20ProcGen%20environment,29)

### Paper notes Joseph Bloom Minigrid and the DT transformer:
In [this](https://www.lesswrong.com/posts/JvQWbrbPjuvw4eqxv/a-mechanistic-interpretability-analysis-of-a-gridworld-agent) paper, Joseph examines MemoryDT, which is a decision transformer trained on the minigrid memory task, where it is shown a specific object and then needs to go and select the correct object from a collection. It's an autoregressive trajectory modelling transformer. This is a very large architectural change from what is going on with the Impala model. 

In this case the model has to cross a gap where it can no longer see the original object and then select the correct object from a selection of options. It only obtains a reward when it chooses correctly. While crossing it needs to hold in its memory what the original object that it saw was.

This makes the model an interesting candidate for interpretability because one would presume the target goal must be held quite clearly in memory, and be identifiable as the model goes about its business. It is also a simple environment which has some significant upsides. Possibly most importantly the nature of the agent is closest to possible LLM style agents because the model is an agent simulator, that simply samples from trajectories that are likely to lead to success. This is similar to how a GPT LLM is a simulator of different characters that can be elicited through prompting or biased with fine-tuning, and what makes them capable general agents.

## Concrete next steps:
Result replication:
- [x] Complete the feature visualisation of the CNN SAE layers. 
- [ ] Replicate whether it's possible to find these directional channels in the network.
	- [ ] This will involve setting up scenarios where the next key is around a little corner in different orientations and see if there is a specific layer that activates more significantly in each case.
- [x] Replicate whether the SAEs are actually as monosemantic as the raw layers. 
	- [x] This will involve just running the feature vis and seeing if the layers are significantly more interpretable in the SAE case. I will also need to double check the paper to see exactly how they reach this result.
- [x] Do feature visualisation on small patches of the channel instead
- [ ] Do an environment dataset image sampling of strongest activating channels in the SAE
	- [x] Do this with patches
	- [ ] Do this with whole images
- [ ] Identify specific things that each SAE channel is responsible for
	- [x] Different directions that need to be taken
	- [ ] different entities to be pursued
	- [x] Explore activation combinations
- [ ] Train up a decision transformer or RNN style network that we can use probes on. 