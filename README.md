# WhyLeNet

"While(){Lenet}"

## Overview

In which we put LeNets inside our LeNet. Every Neuron is a LeNet.

Of course, this poses some problems: 

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

```
def WhyLeNetNeuronPartial(value):
   value = digitToImage(value)
   v = conv2d(value)
   v = maxpool(value)
   # etc
   # v = softmax(v) not required 
   return max_idx(v)


def WhyLenNetNeuron(v,w,x,y,z): 
   a = WhyLeNetNeuronPartial("Hundreds")
   b = WhyLeNetNeuronPartial("Tens")
   c = WhyLeNetNeuronPartial("Ones")
   d = WhyLeNetNeuronPartial("Tenths")
   e = WhyLeNetNeuronPartial("Hundreths")
   N = (100 * a) + (10 * b) + (1 * c) + (0.1 * d) + (0.01 * e)
   return N

```
