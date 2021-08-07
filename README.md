# FaceRecognitionTriplet
Triplet Loss architecture helps us to learn distributed embedding by the notion of similarity and dissimilarity. It’s a kind of neural network architecture where multiple parallel networks are trained that share weights among each other. During prediction time, input data is passed through one network to compute distributed embeddings representation of input data.

## Cost Function
The cost function for Triplet Loss is as follows:
L(a, p, n) = max(0, D(a, p) — D(a, n) + margin)
where D(x, y): the distance between the learned vector representation of x and y. As a distance metric L2 distance or (1 - cosine similarity) can be used. The objective of this function is to keep the distance between the anchor and positive smaller than the distance between the anchor and negative.
