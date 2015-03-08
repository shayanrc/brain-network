function [err grad output]= main()
  nnfunctions;
  inputfunctions;
  
  learningRate=1;
  img=normalize(double(imread("input.jpg")));
  inputMat=makeImgInputs(img,4);

  inputLayer.weights=randn(961,16); %961 neurons with 16 inputs each
  inputLayer.grad=zeros(1,961);
  inputLayer.outputs=zeros(1,961);
  
  layer2.weights=randn(16,961); %16 neurons with 961 inputs each
  layer2.grad=zeros(1,16);
  layer2.outputs=zeros(1,16);
  
  %brainLayer : interconnected layers , each layer has 16 neurons with 3(no of layers)*16 inputs
  brainLayer(1).weights=randn(16,4*16);  
  brainLayer(1).grad=zeros(1,16);
  brainLayer(1).outputs=zeros(1,16);
  brainLayer(2).weights=randn(16,4*16);  
  brainLayer(2).grad=zeros(1,16);
  brainLayer(2).outputs=zeros(1,16);
  brainLayer(3).weights=randn(16,4*16);  
  brainLayer(3).grad=zeros(1,16);
  brainLayer(3).outputs=zeros(1,16);
  
  outputLayer.weights=randn(121,16);
  outputLayer.grad=zeros(1,121);
  outputLayer.outputs=zeros(1,121);
  
  brainLayerIO=zeros(4,16); %matrix containing the output of each brainLayer; also serves as the input for all brainLayers
  
  
  %<forward-loop>
  InputProcessorOutputs=rowInputForward(inputMat,inputLayer.weights,@sigmoid); % forward for first layer
  
  
  brainLayerIO(1,:)=forward (InputProcessorOutputs,layer2.weights,@sigmoid); % forward for layer2
  
  
  
  for i=1:numel(brainLayer)
    
    brainLayerIO(i+1,:)=forward(brainLayerIO(:)',brainLayer(i).weights,@sigmoid)';
    
  endfor
  
  output=forward(brainLayerIO(end,:),outputLayer.weights,@sigmoid);
  desiredOutputs=zeros(size( brainLayerIO(1,:))); %replace with getDesiredOutput
  desiredOutputs(7)=1; %simulated category
  %</forward-loop>
  
  %<error and gradient calculation>
  [err grad]=sigmoidCostFunction (output,desiredOutputs);
  %</error and gradient calculation>
  
  %output=layer2Outputs;
  layer2.grad=grad;
  %imagesc(inputLayer.grad);
  inputLayer.grad=backpropagate(layer2.grad,layer2.weights);
  
  layer2.weights=modifyWeights(learningRate,layer2.grad,layer2.weights,sigmoidDerivative(layer2Outputs),InputProcessorOutputs);
  
  %pause;
  
  %imagesc(inputLayer.grad);
  
  %imagesc(layer2.weights);
  %pause;
  
  
  %imagesc(layer2.weights);
  
endfunction