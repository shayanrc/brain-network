function [err grad output]= main()
  nnfunctions;
  inputfunctions;
  
  learningRate=1;
  %img=normalize(double(imread("input.jpg")));
  %inputMat=makeImgInputs(img,4);
  
  %initialize trainigset
  trSet = getTrainingSet("train"); % get training set from folder
  idx = randperm(length(trSet)); %obtain shuffled indices using randperm
  trainingSet = trSet(idx); %use shuffled indices to obtain shuffled vector

  inputLayer.weights=randn(961,16); %961 neurons with 16 inputs each
  inputLayer.grad=zeros(1,961);
  inputLayer.outputs=zeros(1,961);
  
  layer2.weights=randn(16,961); %16 neurons with 961 inputs each
  layer2.grad=zeros(1,16);
  layer2.outputs=zeros(1,16);

  
  %brainLayer : interconnected layers , each layer has 16 neurons with 3(no of layers)+1 (output of previous layer)*16 inputs
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
  
  
  for trindex = 1:length(trainingSet)

    
    %img=normalize(double(imread("input.jpg")));
    img=normalize(double(imread(trainingSet(trindex).fileDir)));
    inputMat=makeImgInputs(img,4);
    
    disp(sprintf("training on  trainingset for file %s \n",trainingSet(trindex).fileDir));
    
    
    %<forward-loop>
    InputProcessorOutputs=rowInputForward(inputMat,inputLayer.weights,@sigmoid); % forward for first layer
    
    
    brainLayerIO(1,:)=forward (InputProcessorOutputs,layer2.weights,@sigmoid); % forward for layer2
    
    
    %forward for brain layers
    for i=1:numel(brainLayer)
      
      brainLayerIO(i+1,:)=forward(brainLayerIO(:)',brainLayer(i).weights,@sigmoid);
      
    endfor
    
    output=forward(brainLayerIO(end,:),outputLayer.weights,@softmax);
    desiredOutputs=zeros(size( output)); %replace with getDesiredOutput
    desiredOutputs(7)=1; %simulated category
    %</forward-loop>
    
    %<error and gradient calculation>
    [err grad]=sigmoidCostFunction (output,desiredOutputs);
    
    outputLayer.grad=grad;
    %</error and gradient calculation>
   
    %<backpropagation> 
    
    %calculate the backpropagated gradients of the brainlayers separately 
    %since their no. of inputs don't mach outputs of the previous layer
    brainLayerGrad=zeros(4,16);
    brainLayerGrad(4,:)+=backpropagate(outputLayer.grad,outputLayer.weights);
    brainLayerGrad(:)+=backpropagate(brainLayerGrad(4,:),brainLayer(3).weights)';
    brainLayerGrad(:)+=backpropagate(brainLayerGrad(3,:),brainLayer(2).weights)';
    brainLayerGrad(:)+=backpropagate(brainLayerGrad(2,:),brainLayer(1).weights)';
    
    
    
    brainLayer(3).grad=brainLayerGrad(4,:);
    brainLayer(2).grad=brainLayerGrad(3,:);
    brainLayer(1).grad=brainLayerGrad(2,:);
    
    layer2.grad=brainLayerGrad(1,:);
    
    inputLayer.grad=backpropagate(layer2.grad,layer2.weights);
    
    %</backpropagation>
    
    %<modify the Weights>
    %layer2.weights=modifyWeights(learningRate,layer2.grad,layer2.weights,sigmoidDerivative(layer2Outputs),InputProcessorOutputs);
    
    outputLayer.weights=modifyWeights(learningRate,outputLayer.grad,outputLayer.weights,sigmoidDerivative(output),brainLayerIO(4,:));
    
    brainLayer(3).weights=modifyWeights(learningRate,brainLayer(3).grad,brainLayer(3).weights,sigmoidDerivative(brainLayerIO(4,:)),brainLayerIO(:)');
    brainLayer(2).weights=modifyWeights(learningRate,brainLayer(2).grad,brainLayer(2).weights,sigmoidDerivative(brainLayerIO(3,:)),brainLayerIO(:)');
    brainLayer(1).weights=modifyWeights(learningRate,brainLayer(1).grad,brainLayer(1).weights,sigmoidDerivative(brainLayerIO(2,:)),brainLayerIO(:)');
    
    layer2.weights=modifyWeights(learningRate,layer2.grad,layer2.weights,sigmoidDerivative(brainLayerIO(1,:)),InputProcessorOutputs);
    
    inputLayer.weights=modifyRowWeights(learningRate,inputLayer.grad,inputLayer.weights,sigmoidDerivative(InputProcessorOutputs),inputMat);
    
    
    %</modify the Weights>
  
  endfor
  
  %imagesc(inputLayer.grad);
  
  %imagesc(layer2.weights);
  %pause;
  
  
  %imagesc(layer2.weights);
  
endfunction