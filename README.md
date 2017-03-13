# Dynamic Representational Modeling (DRM)

End-to-end training of neural systems.

DRM assume that all components in an artifical brain are neural networks.
That is, we have populations, connections, and readouts that are all specified
in terms of neural networks. These networks can implement very specific
 conduction delays or synaptic mechanisms.

Ideally, DRM allows to identify the circuit diagrams by training a model
 end-to-end.

## Populations

A DRM consists of a set of neuronal populations. Populations can have
physical locations, have measurement units, etc.

NOTE: The stimulus is treated as a population itself
      Implies that the connection can take care of only looking at part of the input

NOTE: The response is treated as a population itself.
      Implies that the connection takes care of specific readouts of e.g. one voxel

## Connections

Stimulus, neuronal populations and readouts are connected to each other
via connections. These connections implement conduction delays, HRFs, etc.
We can think of different variants.

### Fixed delay

A fixed delay connection stores an incoming input for n time steps and
then releases it.

### Memory unit

Delayed response can also be accomplished by a GRU mechanism.
This mechanism allows us to update all populations which
receive input from the stimulus and from each other. While an interconnected
network will not be acyclic, we can just assume that some populations did
not yet receive the input from other units.

### Parameterized function

We can also implement mechanisms that learn the conduction delay or shape of the
conduction delay.

## Readout

A readout is linked to one or more populations via connections. It combines
all population responses and internally can separate them (e.g. voxel-specific
bold responses) or combine them (e.g. MEG sensor readings). It is a
CMS model object (like a regressor).

## TO DO

* make iterator work seamlessly
* also make the links between populations neural networks; this also
allows implementation of synaptic delays. We actually need this! If we
have a mechanism to generate delays then models with cycles become acyclic!
In contrast, the readout mechanism can be instantaneous
All of this does require that we sample at very high rates... Maybe we
can allow the connection to learn the delay. Any >0 delay will do...
* Can we use DRM to reverse engineer circuits in systems neuroscience? Think Ganguli, Bethge, etc.
* Allow each population to connect to multiple outputs
* Add documentation!
* Circuit analysis of the models we learn; link to mechanisms
* Can we just use one huge RNN? This is a subset of our DRM approach
* First work on simulations!
* incorporate synaptic mechanisms (build models for this)
* First problem to fix: how to implement a learnable or chosen conduction delay? tapped delay lines?
Can we just push hidden states on a FIFO stack?
* build learnable examples of readout mechanisms for meg, fmri, calcium imaging,  firing rate, spiking
* and add cost functions on top
* we can also build a plausible connectivity graph and then condition categorization on IT
* replace n_output with output size everywhere to handle image data, convolutional output, etc