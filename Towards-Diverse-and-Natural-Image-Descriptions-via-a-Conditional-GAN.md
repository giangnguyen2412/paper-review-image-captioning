# Towards Diverse and Natural Image Descriptions via a Conditional GAN - Bo Dai et al, `ICCV 2017`

## Novelties

- This paper is the first time we use GAN method in image description, relying on the conditional GAN to learn the generator, instead of using MLE (we always compare the labels and predicton).

- Not only resulting in a generator, but also yielding a desciption evaluator at the same time, which is subtially more consistent with human evaluation (see experiments).

## Challenges
Despite of potentials from using GAN for image captioning, there remains challenges as below:

- **First**, in contrast to image generation, where the transformation from the input random vector to the produced image is a deterministic continuous mapping, the process of generating a linguistic description is a **sequential sampling** procedure. Its non-differentiable, making it difficult to apply back-prop directly.

- **Second**, in conventional GAN, when finishing the generation process, the discriminator will work and give feedback to the generator. For sequence generation, this would lead to several difficulties in training, including gradient vanishing and error propagation.


## Related work
### Generation
Generating descriptions for images has been a long standing topic in computer vision.

- Detection-based: First detect the visual concepts using CRF, SVM or CNNs, then generate descriptions using templates or by retrieving relevant sentences from existing data.

- Encoder-Decoder: Given an image I, it first derives a feature representation ***f(I)***, and then generates the word *w1, w2, ... wT* sequentially, following a Markov process conditioned on ***f(I)***. The model paramters are learned via maximum likelihooh estimation MLE.

- Neural Network: Use CNN for deriving the visual features ***f(I)***, and an LSTM net to express the sequential relations among words.

### Evaluation
Various evaluation metrics have been proposed

- BLEU and ROUGE: focus on the precision and recall of n-grams. 

- METEOR: combine both recall and precision of n-grams.

- CIDEr: use weighted statistics over n-grams

- SPICE: instead of matching between n-grams, it focuses on those linguistic entities that reflect visual concepts, however naturalness of the expressions are not considered.

## Framework
### Overall formulation
![](https://github.com/luulinh90s/paper-review-image-captioning/blob/master/images/Towards-Diverse-and-Natural-Image-Descriptions-via-a-Conditional-GAN/Untitled.png)

Here the random vector z allows the generator to product different description given an image. One can control the diversity y tuning the variance of z (z can changes in a wide range). 

The evaluator E is also a NN with similar architecture to G but operating in a different way. Given an image I and a descriptive sentece S, it embeds them into vector ***f(I)*** and ***h(S)*** of **the same dimension**. Then, the quality of the description is measured by the dot product of the embedded vectors. We apply a sigmoid function to turn the result into a prob. value in [0, 1]

### Training G: Policy Gradient and Early Feedback
The process of generation is non-differentiable (here I wanna explain what is differentiable and non-differentiable which appears a lot in ML. A process is differentiable is when each output does not depend on others, the generation of outputs are digtinguishable. And non-differntiable means the output has dependence on others, so when we apply back-prop, we need to traverse over the dependence chain which can lead in gradient vanishing).

To deal with this, we use *Policy Gradient* from Reinforcement Learning. The basic idea is to consider a sentence as a sequence of actions, where each word *wt* is an action. The choices of such "actions" are governed by a policy **πθ**. I actually dont know much ab RL, then I will leave here and may come back to explain in details later on.
### A tutorial for Monte Carlo method is needed!!
### Training E: Naturalness and Relevance
A good description needs to satisfy two criteria: natural and semantically relevant. To enforce both criteria, we consider three types of descriptions for each training image I : 

(1) *SI* : the set of descriptions for I provided by human, 

(2) *SG*: those from the generator ***Gθ***

, and (3) *S\I* : the human descriptions for different images, which is uniformly sampled from all descriptions that are not associated with the given image I . To increase the scores for the descriptions in *SI* while suppressing those in the others, we use a joint objective formulated as:

![](https://github.com/luulinh90s/paper-review-image-captioning/blob/master/images/Towards-Diverse-and-Natural-Image-Descriptions-via-a-Conditional-GAN/Untitled1.png)

Here the first term is for the desciptions associated with the given image. Then the value of the first term should be large. The second term is to force the evaluator to distinguish between the human descriptions and the generated ones, which would in turn provide useful feedback to ***Gθ*** (to help Generator in back-propagation through evaluating quality of descriptions), pushing it to generate more netural descriptions.

The third term ensures the *sematic relevance*, by explicitly suppressing mishmatched descriptions. Because the descriptions are not associated with the given image, adding this can keep generator from generating mismatch desciption given the image.

The *alpha* and *beta* are to balance the contributions of these terms, whose values are empirically determined on the validation set.

## Experiments
