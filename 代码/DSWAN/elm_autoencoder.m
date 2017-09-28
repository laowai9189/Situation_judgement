function [OutputWeight] = elm_autoencoder(InputData, NumberofHiddenNeurons)
%ELM_AUTOENCODER Summary of this function goes here
%   Detailed explanation goes here
InputData = InputData';
NumberofTrainingData = size(InputData, 2);
NumberofInputNeurons = size(InputData, 1);
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*InputData;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;
H = 1 ./ (1 + exp(-tempH));
OutputWeight=(pinv(H') * InputData')'; 
end

