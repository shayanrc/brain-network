function [err grad output]= main()
  nnfunctions;
  inputfunctions;
  
  learningRate=1;
  img=normalize(double(imread("input.jpg")));
  inputMat=makeImgInputs(img,4);

  inputLayer.weights=randn(961,16); %961 neurons with 16 inputs each
  inputLayer.grad=zeros(1,961);
  
  layer2.weights=randn(16,961); %16 neurons with 961 inputs each
  layer2.grad=zeros(1,16);
  
  %brainLayer : interconnected layers , each layer has 3 neurons with 3*16 inputs
  brainLayer(1).weights=randn(6,6*16);  
  brainLayer(1).grad=zeros(1,6);
  brainLayer(2).weights=randn(6,6*16);  
  brainLayer(2).grad=zeros(1,6);
  brainLayer(3).weights=randn(6,6*16);  
  brainLayer(3).grad=zeros(1,6);
  
  outputLayer.weights=randn(121,16);
  outputLayer.grad=zeros(1,121);
  
  %brainLayerIO; %matrix containing the output of each brainLayer; also serves as the input for all brainLayers
  
  
  
  InputProcessorOutputs=rowInputForward(inputLayer.weights,inputMat,@sigmoid);
  
  
  layer2Outputs=forward (layer2.weights,InputProcessorOutputs,@softmax);
  
  
  desiredOutputs=zeros(size(layer2Outputs));
  %errors=zeros(1,961);
  [err grad]=sigmoidCostFunction (layer2Outputs,desiredOutputs);
  output=layer2Outputs;
  layer2.grad=grad;
  %imagesc(inputLayer.grad);
  inputLayer.grad=backpropagate(layer2.grad,layer2.weights);
  %pause;
  
  %imagesc(inputLayer.grad);
  
  %imagesc(layer2.weights);
  %pause;
  
  layer2.weights=modifyWeights(learningRate,layer2.grad,layer2.weights,sigmoidDerivative(layer2Outputs),InputProcessorOutputs);
  
  %imagesc(layer2.weights);
  
endfunction