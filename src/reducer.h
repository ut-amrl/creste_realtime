/**
 * This class takes in inputs feature torch tensors and compute the dimensionality reduction
 * using PCA. It can be used to return the reduced feature tensor map for the current batch.
 * 
 * It is designed to keep computing the PCA or the received feature tensors until it has received L
 * feature tensors. Then it drops the oldest feature tensor and computes the PCA for the remaining
 * feature tensors.
 * 
 */

class Reducer {

};