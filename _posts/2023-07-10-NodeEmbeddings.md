---
title: Node Embeddings
date: 2023-07-10 10:00:00 +0800
categories: [Machine Learning, Graph]
---
This blogpost acts as a note to the [Machine Learning with Graphs course](http://web.stanford.edu/class/cs224w/).
## Idea
In graph theory, we often encounter high-dimensional, non-Euclidean data structures. While such graph data encapsulates intricate relationships and connectivity patterns, it is not directly amenable to traditional machine learning algorithms that require structured, tabular, and often Euclidean data. 

This is where **node embeddings** come into play. Node embeddings transform the nodes of a graph into a lower-dimensional, continuous vector space, aiming to maintain the graph's structural properties. This transformation is akin to a function 
$$
f: V \rightarrow \mathbb{R}^d,
$$
 mapping each node in the graph \( V \) to a \(d\)-dimensional real-valued vector. The resulting vector space is often referred to as the *embedding space*.

The similarity of the embedding vectors in the embedding space is indicative of the similarity of the corresponding nodes in the original graph. This means that nodes that are close to each other in the embedding space are also close to each other in the original graph. Note that there are different definitions to the notion of similarity or "closeness". The ultimate goal is to encode the structural properties of the graph into the embedding space and potentially use it for downstream tasks such as node classification, link prediction.

## Node Embeddings 
To make the definition of node embeddings more concrete. Given a graph \(G = (V, E)\), a node embedding is a function \( f: v_i \rightarrow y_i \in \mathbb{R}^d \) \( \forall i \in [n] := \{ 1,2,\dots,n \} \) where \(n\) is the number of nodes in the graph such that \( d \ll n \). We can also say that \( n \) is the "dimension" of the graph even though it is not a vector space.

### Encoder-Decoder Framework
Given the goal of encoding nodes such that the **similarity in the embedding space approximates the similarity in the original graph**. Once again, we are letting the "similarity" be an abstract metric of two nodes. Assuming in the embedding spaces, all the embed vectors are normalized to have unit length, we can define the similarity between two nodes \( u\) and \( v \) as the dot product of their corresponding embedding vectors \( z_u \) and \( z_v\), i.e,
$$
similarity(v_i, v_j) \approx z_u^T z_v. 
$$
This means that the similarity between two nodes is approximately the cosine similarity between their embedding vectors. 

In this framework, the **encoder** acts as the embedding function which "encodes" or maps a node to the embedding space. The **decoder** acts as the similarity function which "decodes" or maps the embedding vectors to a similarity score. This framework is quite powerful and abstract in the sense that it provides a general strategy for tackling a wide range of complex problems, not necessarily limited to node embeddings. Its core idea is quite intuitive and simple: encode the original data into a lower-dimensional space and then to assess the performance of the encoder, we decode the encoded data back to the original space and compare it with the original data and see how well it performs. If the decoded data is close to the original data, then the encoder is doing a good job and vice versa. 

For example, lets imagine a game with two people A and B where A is given a word and has to give a drawing to B such that B can guess the word. In this game, A is the encoder and B is the decoder. A's goal is to encode the word into a drawing such that B can guess the word. B's goal is to decode the drawing into the word that A was given. If B can guess the word correctly, then A is doing a good job and vice versa.

### Shallow Encoding
This is one of the simplest encoding approach. It only looks up the embedding vector given a node: 
$$
 \textbf{ENCODE}(v) = Z \cdot 1_v
$$
where \( Z \in \mathbb{R}^{d \times |V|} \) is the embedding matrix and \( 1_v \in \mathbb{I}^{|V|} \) is the indicator vector, zeroes everywhere except a one at the index corresponding to the node \( v \). This means that the embedding vector of a node is the corresponding column of the embedding matrix. This learning method usually starts by defining the encoder, the similarity function, and then optimize the encoder such that 
$$
similarity(u,v) \approx z_u^T z_v.
$$
This approach has a few major drawbacks:
1. The size of the encoder is \( \textbf{O}(|V|)\) which has major impact on scalability for large graphs such as social networks.
2. It is not easy to define the similarity metric.  
3. It only works for nodes already present in the graph thus making it unusable for unseen nodes. 

Thus, this method is not very good in practice. 

### Random Walks
Given a graph and a starting point, we randomly select a neighbor, and move to this neighbor; then we select a neighbor of the current point at random and move to it, etc. The random sequence of nodes we visit is called a random walk.