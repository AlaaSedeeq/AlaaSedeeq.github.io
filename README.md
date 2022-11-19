
theme: jekyll_plugins

# Chapter 5: Neural Network (Summary)

## GLM
- $y = f(W^{T}.\Phi(x))$ , where $f$ is any monotonous and differentiable function (link function)
    - In Regression, f is the identity.
    - In Classification, f is a non-linear activation function.
- Are much more transparent than NNs in their operation.
- Have useful analytical and computational properties.
- Limit you to certain categories of models whereas theoretically the range of functions that can be represented by NNs is unlimited.
- Thier applicability is limited by the curse of dimensionality(No. of $\Sigma$ and $\mu$ for the features)<br><br>
    
## Neural Network
- NN uses adaptive basis functions instead of fixed basis functions(e.g. GLM) to be applicable in largescale problems.
<img src='Results/FFNN.jpg' width='500' height='200'></img>
- No. of basis functions is fixed in advance, but allow them to be adabtive to the data.
- So, $\Phi_{j}$ havee their $W_{ji}$, and $W_{ji}$ are adaptive to the during training.
- Most successful model of NN is the Feed-Forward Neural Network(FFNN).
- Each basis function(neuron) is a non-linear function of a linear combination of the input.
- NN can be represented as a series of functional transformation.
- $y_{k}(X, W) = h^{(N)}(a^{(N)}(h^{(N-1)}(.....(a^{(1)}(X)).....)))$
- A stacked linear combination followed by non-linear activations.
- Number of layers equals number of layers with adaptive weights.

### NN Training
- NN is viewed as a parametric nonlinear function
- To determine its parameters, minimize some error function(sum-of-squares).
    - $E(W) = \frac{1}{2}\sum_{n=1}^{N}\|y(x_{n}, W) - t_{n}\|$
- No hope of analytical solution, so it's solved numerically.
- we want to find W such that E(W) is smallest, but E(W) has a highly nonlinear dependence on $W \& b$, an dthere will be many points where the $\nabla_{W} E(W)$ vanishes.
- The minimum corresponding to the smallest $E(W)$ is the global minimum.
- Other minima with higher $E(W)$ is local minimum.
- The optization problem for the continuous non-linear function can be solved by several techniques, most of them involves initializing $W^{0}$, then moving through $W$-Space in a succession step
    - $W^{(\tau+1)} = W^{(\tau)} + \Delta W^{(\tau)}$

1. #### Gradient Descent optimization
    - The simplest appproach to using gradient information
    - Take small steps in the direction of $-\nabla E(W)$
        - $W^{(\tau+1)} = W^{(\tau)} - \eta \nabla E(W^{(\tau)})$, where $\eta$ is the learninig rate and $\eta$ > 0
        - After each update the Gradient is re-evaluated for the new $W$ and the process is repeated.
        - The error function is defined w.r.t a training set, so each step requires that the entire training set be processed in order to evaluate $\nabla E(W)$
        - Techniques that use the whole data set at once are called batch methods.
2. #### Stochastic Gradient Descent optimization
    - Useful for large data set.
    - On-line gradient descent, also known as sequential gradient descent, makes an update to the weight vector based on one data point at a time, so that:
        - $W^{(\tau+1)} = W^{(\tau)} - \eta \nabla E_{n}(W^{(\tau)})$
    - This done by cycling through the data in sequential or by sampling from it.

3. #### Mini-batch optimization
    - Intermediate methods, one sample < batch size < number of data points
    
### Regularization
- It's any modification to a learning algorithm to reduce its generalization error, but not training error.
- Goals for regularizer:
    - Encode prior knowledge
    - Express preference for simple model
- Baysian POV is that the regularization corresponds to prior distributions on model parameters
- Generalization error is not a function of number of neurons, due to the presence of local minima in the error function.
- Alternatively, choose large number of neurons, and control the complexity by a regularization term.
- Any valid regularization needs to invariant to scale and shift.
1. Simple weight decay regularizer 
    - $\tilde{E(W)} = E(W) + \frac{\lambda}{2}W^TW$
    - Does not meet this property
    - It will treat all $W$ on equal footing, whereas to get the same output with transformed date we need each $W$ to be treated differently.
- To be valid, we need each $W$ to have a different regularization term
    - $\tilde{E(W)} = E(W) + \frac{\lambda_{1}}{2}\sum_{w\in{W_{1}}}{W^TW}+ \frac{\lambda_{2}}{2}\sum_{w\in{W_{2}}}{W^TW}$ + ...
    - It's equivalant to a prior for the NN.[See](https://github.com/AlaaSedeeq/Pattern-recognition-and-machine-learning/blob/main/Ch%2003%20Linear%20Models%20For%20Regression/00%20Chapter%20Summary.ipynb)
    - But it's not a proper prior (can not be normalized), and leads to difficulties in selecting regularization coefficient and model comparasion with bayesian framework because the corresponding evidence is zero.
2. Early stopping
- Alternative to regularization, it can control model complexity.
- Stop learning validation error starts to increase.

### Invariance
- We need the rediction to be unchanged, or invariant under one or more transformations of the input.
- Approaches for the adaptive model:
1. Training set augmentation:
    - Transform data to the desired invariance.
    - Easy to implement,and can encourage complex invariance.
    - But computational cost.
2. Tangent propagation:
    - A regularization term is added to $E(W)$ that penalizes changes in the model outpus, when the input is transformed.
    - $\tilde{E}(W) = E(W) + \lambda \Omega$,  $\Omega = \frac{1}{2}\sum_{n}\sum_{k}(\sum^DJ_{nkj}\tau_{nj})^2$
3. Convolution neural network
    - Build the invariance property in the NN structure.
    - This approach exploits the fact that nearby pixels in an image are strengly correlated than distance pixels.
    - The neurons are organized into planes, each is called feature map.
    - Incorporate three mechanisms:
        - <b>Local receptive fields</b>
            - Each neuron in the same feature map takes input from a small subregion of the input.
            - Extract local features that depend only on small subregions of the image, and such features can be merged in later processing stages to detect higher-order features.
        - <b>Weight sharing</b>
            - $W$ are connected to small region of the input.(Not fully connected)
            - All neurons in the same feature map detect the same pattern , but from different locations in the input.
        - <b>Sub-sampling</b>
            - sub-sampling layers perform a local averaging, reducing the resolution of the feature map, and reducing the sensitivity of the output to shifts and distortions. 
            
4. Soft-weight sharing
- One way of reducing complexity of NN with large number of parameters.
- Encourages group of $W$ to have similar values.
- In CNN, Hard-weight sharing, it constrains groups to be equal, but this can be applied when form of the network can be specified in advance.
- In simple weight-decay, the negative log of a gaussian prior distribution over $W$, but in soft-weight sharing, assume the prior is mixture of gaussians.
    - $P(W) = \prod_{i}P(W_{i})$
    - $P(W_{i}) = \sum_{j=1}^M \pi_{j} N(w_{j} | \mu_{j}, \sigma_{j}^2)$
    - $\Omega(W) = -\sum_{i} ln(\sum_{j}^M\pi_{j}N(w_{j} | \mu_{j}, \sigma_{j}^2))$
    - $\tilde{E}(W) = E(W) + \lambda \Omega(W)$
    - Then minimize $\tilde{E}(W)$ w.r.t { $\pi_{i}, \mu_{j}, \sigma_{i}$ }
- The learning process soft-weight sharing determines:
    - Division of $W$ into groups($\pi$).
    - Mean weight value for each group($\mu$).
    - Spread of values($\sigma^2$).
