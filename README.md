# WhyLeNet

"While(){Lenet}"

## Overview

In which we put LeNets inside our LeNet. Every Neuron is a LeNet.

This is true "Lenet in Lenet" unlike papers like ResNet in ResNet which are deceptively named (https://arxiv.org/abs/1603.08029)

Note: we use LeNet-1 instead of Lenet-5 as it will be 2,578 instead of 60,000 parameters. It also helps to alleviate overfitting from our large amount of neurons (150,000 vs 3,000,000)

Of course, this still poses some problems: 

## The Input and Output of Lenet are not neurons

Indeed. But they do share a common property - the input is handwritten images of digits 0-9 and the output is a vector with classification predictions for each index 0-9
Therefore, we add an extra operation to map the highest probability index to a handwritten number from the dataset. for simplicity we reuse the same ones (TODO: averaging? strongest prediction?)

## But the value of a neuron isn't 0-9!

Right, so we have to change our implementation. we remove the softmax layer and instead output a single neuron that we will convert to handwritten digit instead

## But it's still not going to be 0-9
Right, and while we might be able to hack this so the network is effectively 3 and a bit quantized, it is a safer bet to continue to use our existing datatype.

So we actually need to add **more** lenets. One for each unit. (100s, 10s, 1s, 0.1s, 0.001s) So now we have five LeNets per neuron. We then add a summation after these networks run and clamp to the float range

Future research could look at 8bit variants which would reduce our LeNets by 2.

## Structure

For each neuron, we replace with: 

```python

DIGIT_BANK = load_mnist_clearest_images() 
DIGITS = torch.arange(10, dtype=torch.float32) # [0, 1, 2, ..., 9]

def digitToImageDifferentiable(prob_distribution):
    """
    Takes a 10-dimensional probability vector and blends the Digit Bank.
    Input shape: [Batch, 10]
    Output shape: [Batch, 1, 28, 28] (A clean or ghosted handwritten image)
    """
    # Einstein summation performs the smooth, continuous matrix blend
    return torch.einsum('bi,ichw->bchw', prob_distribution, DIGIT_BANK

)
def WhyLeNetNeuronPartial(value):
   value = digitToImage(value)
   v = conv2d(value)
   v = maxpool(value)
   # etc
   v = softmax(v) # maybe not required?
   # v = argmax(v)
   # we can't use argmax because it will kill the gradient descent. but by doing some math, the below should operate similarly: 
   v = torch.sum(v * DIGITS.to(logits.device), dim=-1)
    
   return v


def WhyLenNetNeuron(v,w,x,y,z): 
   a = WhyLeNetNeuronPartial("Hundreds")
   b = WhyLeNetNeuronPartial("Tens")
   c = WhyLeNetNeuronPartial("Ones")
   d = WhyLeNetNeuronPartial("Tenths")
   e = WhyLeNetNeuronPartial("Hundreths")
   N = (100 * a) + (10 * b) + (1 * c) + (0.1 * d) + (0.01 * e)

   dist_a = get_unit(N, 100)
   dist_a = get_unit(N, 10)
   dist_a = get_unit(N, 1)
   dist_a = get_unit(N, .1)
   dist_a = get_unit(N, .01)

   # Re-synthesize 5 BRAND NEW handwritten 28x28 images using your einsum trick
    img_a = digitToImageDifferentiable(dist_a)
    img_b = digitToImageDifferentiable(dist_b)
    img_c = digitToImageDifferentiable(dist_c)
    img_d = digitToImageDifferentiable(dist_d)
    img_e = digitToImageDifferentiable(dist_e)
    
    # Stack them together into a 5-channel image block [Batch, 5, 28, 28]
    output_images = torch.cat([img_a, img_b, img_c, img_d, img_e], dim=1)
    
    # Return BOTH the images for the next hidden layer, and N for tracking/final loss
    return output_images, N

   return N

def WhyFC():
   # To-do 

def WhyLenet(inputimg):
   # Todo : convs, etc
   a = Conv(inputimg)
   a = ....
   a = WhyFC()
   return a

```

note we must make the get unit differentiable too:

```Python
def get_unit(unit_scalar, position_weight):
    """
    Differentiable unit extraction.
    If position_weight is 10, it extracts the tens place from N.
    """
    # 1. Isolate the target single digit (e.g., if N=42.15, for weight 10, target is 4.215)
    # We use a soft shift rather than hard modulo math
    target_digit = unit_scalar 
    
    # 2. Compute absolute distance to each of the 10 static digit possibilities [0..9]
    distances = torch.abs(target_digit.unsqueeze(-1) - DIGITS.to(unit_scalar.device))
    
    # 3. Softmin turns closest distance into the highest probability distribution
    # The scaling factor (10.0) controls how strictly sharp the resulting digit lookups are
    prob_distribution = torch.softmax(-distances * 10.0, dim=-1)
    
    return prob_distribution

```
