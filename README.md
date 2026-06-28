# WhyLeNet

In which we put LeNets inside our LeNet. Every Neuron is a LeNet.

Of course, this poses some problems: 

## The Input and Output of Lenet are not neurons

Indeed. But they do share a common property - the input is handwritten images of digits 0-9 and the output is a vector with classification predictions for each index 0-9
Therefore, we add an extra operation to map the highest probability index to a handwritten number from the dataset. for simplicity we reuse the same ones. 

## But the value of a neuron isn't 1-10!
