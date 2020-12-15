# CS 245
## Project introduction

**Project 5: Dynamic Graph Neural Networks For Monitoring Pandemics And Risk Factors**.  
Description:  
  Geographical proximity plays a predominant role in virus spread. An outbreak in one location
  often leads to outbreaks in its surrounding regions (often with a lag time). Knowledge gained
  from an early outbreak may help to provide more precise forecasts for the outbreaks in
  neighboring regions.
  To model this spatial spread, locations of different granularities (e.g., county, state, country) are
  represented by separate nodes with edges in between denoting their relationships (e.g.,
  is-part-of, adjacent). To be specific, two states are considered neighbors and connected by an
  edge if they share a border. Likewise, we model counties and their corresponding adjacency
  relations in a similar fashion. A county is also connected to its state by an “is-part-of” edge.
  These two levels of location nodes and their edges do not vary over time and thus constitute the
  backbone of the dynamic graph. Each location node has associated a set of static attributes
  derived from the Census data and a set of dynamic attributes (such as weather condition,
  number of new cases, number of recovered, and number of new deaths) that evolve over time.
  Given a dynamic graph, representation learning aims to learn informative node representations
  over time by encoding both temporal evolution patterns and structural neighborhood
  information. They can serve as powerful tools for monitoring pandemics and identifying risk
  factors by considering self-evolution, influence from adjacent regions, as well as impacts from
  the other risk factors such as weather. We ask you to implement a dynamic graph neural
  network (DGNN) that views the dynamic graph as a whole by adding inter-time propagation
  edges. The introduction of inter-time propagation edges expands node neighbors along the
  temporal dimension, which allows DGNN to model structural temporal dependencies among
  nodes without an RNN. To further speed up the learning, we can set a window along the
  temporal dimension and only consider inter-time propagation edges within the time window.
  DGNN can be viewed as a special case of Graph Neural Network (GNN) characterized by the
  information propagation equation. The difference is that every node is associated with a
  specific timestamp in DGNN. For pandemics prediction, use the neural ordinary differential
  equations, where we input current location embeddings (e.g. Los Angeles at Mar.8) and predict
  its latent representation at the next timestamp (e.g. Los Angeles at Mar.9). This approach is a
  natural extension of traditional epidemic models, e.g., SIR, that utilize hand-crafted ordinary
  differential equations to model disease spread. Here we utilize a neural network to
  automatically learn ordinary differential equations from data. This model can be easily
  parallelized and adapted as the dynamic graph evolves.
  
Dataset:  
    - SafeGraph Mobility data, CDC pandemic trackers.   
Input:  
    - Graph structure data (node feature and adjacency matrix).    
  
## Team Ideas
Chenyang Wang, Danfeng Guo, Haochen Yin, Huiling Huang,  Panqiu Tang, Yijing Zhou
