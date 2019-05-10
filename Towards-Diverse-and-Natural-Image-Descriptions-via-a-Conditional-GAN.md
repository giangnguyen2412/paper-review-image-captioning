# Towards Diverse and Natural Image Descriptions via a Conditional GAN - Bo Dai et al, `ICCV 2017`

## Novelties

- This paper is the first time we use GAN method in image description, relying on the conditional GAN to learn the generator, instead of using MLE (we always compare the labels and predicton).

- Not only resulting in a generator, but also yielding a desciption evaluator at the same time, which is subtially more consistent with human evaluation (see experiments).

## Challenges
Despite of potentials from using GAN for image captioning, there remains challenges as below:

- **First**, in contrast to image generation, where the transformation from the input random vector to the produced image is a deterministic continuous mapping, the process of generating a linguistic description is a **sequential sampling** procedure. Its non-differentiable, making it difficult to apply back-prop directly.

- **Second**, in conventional GAN, when finishing the generation process, the discriminator will work and give feedback to the generator. For sequence generation, this would lead to several difficulties in training, including gradient vanishing and error propagation.
