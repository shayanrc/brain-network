%script file containing NN functions
1; 

%==============Activation functions==============%

function g = sigmoid(z)
% sigmoid Compute sigmoid function
% J = SIGMOID(z) computes the sigmoid of z.
% z can be a scalar, a vector, or a matrix.
g = zeros(size(z));
g = 1 ./ (1 + exp(-z));
endfunction

function mu = softmax(eta)
  s=sum(exp(eta));
  mu = exp(eta)./s;
endfunction


%==============Derivatives of Activation functions==============%

function x=sigmoidDerivative(y)

  x=y.*(1-y);
endfunction

%==============Cost functions==============

function [cost partGrad]= sigmoidCostFunction(outputs,desiredOutputs)
%sigmoidCostFunction: function to calculate cost function and part of the gradient for sigmoid activation
%outputs : outputs of the network
%desiredOutputs : desired Outputs of the network
cost =(-desiredOutputs.* log(outputs) - (1 - desiredOutputs).* log(1-outputs));
partGrad=desiredOutputs-outputs;
endfunction


%==============Forward propagation functions==============

function outputs =forward(inputs,weights,activation)
% forward: generic forward propagation function
% inputs: input vector to the layer 
% weights: the weights matrix (no. of rows = no of neurons in layer & no. of columns = no. of inputs)
% no of columns of inputs and weights must be equal  outputs=weights*inputs';
  outputs=weights*inputs';
  outputs=activation(outputs)';
endfunction

function outputs = rowInputForward(inputs,weights,activation)
% rowInputForward: forward propagation function where each row of input matrix is fed to one neuron
% inputs: input vector to the layer 
% weights: the weights matrix (no. of rows = no of neurons in layer & no. of columns = no. of inputs)
% no of columns of inputs and weights must be equal
  outputs=forward(inputs,weights,activation);
  outputs=sum(outputs.*eye(size(outputs)));
endfunction


%==============Backpropagation functions==============
%these function(s) propagate the error/gradient backwards
%note: this function should be called for a layer before modifyWeights
function prevLayerGrad=backpropagate(partGrad,weights)
 %prevLayerGrad=partGrad*weights; %Original
prevLayerGrad=partGrad*weights;
endfunction

%==============Modify Weights functions==============
%

function changedweights = modifyWeights(learningRate, costDerivative, weights, activationDerivative, inputs)
  %modifyWeights : function to modifiy weights
  %learningRate : learning rate of the network (scalar)
  %weights : weights matrix of a layer
  %partGrad : derivative of the cost function / propagated error
  weightChanges=learningRate*(costDerivative.*activationDerivative);
  %weightChanges=activationDerivative.*costDerivative';
  %weightChanges*=learningRate;
  weightChanges=weightChanges'*inputs; %Original
  %weightChanges=weightChanges*inputs;
  
  %colorbar;
  %imagesc(weights)
  
  changedweights=weights+weightChanges;
  
  %imagesc(weights);
endfunction

function changedweights = modifyRowWeights(learningRate, costDerivative, weights, activationDerivative, inputs)
  %modifyRowWeights : function to modifiy weights for convulational layer
  %learningRate : learning rate of the network (scalar)
  %weights : weights matrix of a layer
  %partGrad : derivative of the cost function / propagated error
  
  %weightChanges=learningRate*(costDerivative.*activationDerivative);
  %weightChanges=weightChanges'*inputs; %Original
  
  temp=costDerivative.*activationDerivative;
  weightChanges=zeros(size(weights));
  
  for i=1:(size(weights)(1))
    weightChanges+=temp'*inputs(1,:);
  endfor
  
  changedweights=weights+weightChanges;
  
  %imagesc(weights);
endfunction