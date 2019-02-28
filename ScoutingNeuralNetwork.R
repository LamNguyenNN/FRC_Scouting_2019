initWeightMats = function(topo) {
  i = 1
  weightMats = list()
  while(i < length(topo)) {
    weightMats[[i]] = matrix(data = rnorm(n=topo[i]*topo[i+1], mean=0, sd=1), nrow=topo[i], ncol=topo[i+1], byrow=T)
    i = i + 1
  }
  return (weightMats)
}

initBiasMats = function(topo, numInputSets) {
  i = 1
  biasMats = list()
  while(i < length(topo)) {
    biasMats[[i]] = matrix(data = rnorm(n=topo[i+1], mean=0, sd=1), nrow=numInputSets, ncol=topo[i+1], byrow=T)
    i = i + 1
  }
  return (biasMats)
}

sigmoid = function(x) {
  1 / (1 + exp(-x))
}

elu = function(x) {
  ifelse(x > 0, x, eluAlpha * (exp(x) - 1))
}
eluDerivative = function(x) {
  ifelse(x > 0, 1, eluAlpha * exp(x) )
}

relu = function(x) {
  ifelse(x > 0, x, 0)
}
reluDerivative = function(x) {
  ifelse(x > 0, 1, 0)
}

tanhDerivative = function(x) {
  return (1 - tanh(x)^2)
}

softmax = function(x) {
  sum = sum(exp(x))
  exp(x)/sum
}

forwardProp = function(inputMat, weightMats, biasMats) {
  numSynapses = length(weightMats)
  
  activatedSums = list()
  
  activatedSums[[1]] = relu((inputMat %*% weightMats[[1]]) + biasMats[[1]])
  
  i = 2
  while(i < numSynapses) {
    activatedSums[[i]] = relu((activatedSums[[i-1]] %*% weightMats[[i]]) + biasMats[[i]])
    i = i + 1
  }
  
  output = (activatedSums[[numSynapses-1]] %*% weightMats[[numSynapses]]) 
  + biasMats[[numSynapses]]
  
  unactOutput = output
  
  for(i in 1:nrow(output)) {
    output[i,] = softmax(output[i,])
  }
  
  return (list("activatedSums" = activatedSums, "output" = output, "unact" = unactOutput))
}

MSECost = function(targetOutput, netOutput) {
  error = (1/nrow(targetOutput)) * sum( (1/ncol(targetOutput))*(rowSums((targetOutput - netOutput) ^ 2)) ) return(error)
}

CrossEntropyCost = function(targetOutput, netOutput) {
  error = (1/nrow(targetOutput)) * sum( rowSums(-targetOutput * log(netOutput)))
  return(error)
}

calcAccuracy = function(output, trainOutput) {
  numCorrect = 0
  for(i in 1:length(output)) {
    if (output[i] == trainOutput[i]) {
      numCorrect = numCorrect + 1
    }
  }
  accuracy = numCorrect / length(trainOutput)
  return (accuracy)
}

SGD = function(inputMat, weightList, biasList, outputList, targetOutput, learningRate, epoch, input_test, output_test) {
  origInput_mat = inputMat
  origOutput_mat = targetOutput
  synapseIndex = length(weightList)
  epochNum = 1
  counter = 0
  prevCost = 0
  currCost = 0
  
  deltaWeightList = list()
  gradWeightList = list()
  gradBiasList = list() #Gradient is same as delta for bias
  
  for(i in 1:synapseIndex) {
    deltaWeightList[[i]] = matrix(nrow=nrow(weightList[[i]]),
                                  ncol=ncol(weightList[[i]]), byrow = T)
    gradWeightList[[i]] = matrix(nrow=nrow(weightList[[i]]),
                                 ncol=ncol(weightList[[i]]), byrow = T)
    gradBiasList[[i]] = matrix(nrow=nrow(biasList[[i]]), 
                               ncol=ncol(biasList[[i]]), byrow = T)
  }
  
  while(T) {
    for(trainEx in 1:nrow(targetOutput)) {
      outputList = forwardProp(inputMat, weightList, biasList)
      if(anyNA(outputList$output)) {
        print(outputList$activatedSums)
        print(outputList$unact)
        print(outputList$output)
        print("SGD")
        stop()
      }
      
      delta = outputList$output[trainEx,] - targetOutput[trainEx,]
      
      deltaWeightList[[synapseIndex]] = matrix(delta, 
                                               nrow=nrow(weightList[[synapseIndex]]), 
                                               ncol=ncol(weightList[[synapseIndex]]), byrow = T)
      
      for(i in 1:ncol(outputList$activatedSums[[synapseIndex-1]])) {
        gradWeightList[[synapseIndex]][i,] = deltaWeightList[[synapseIndex]][i, ] * outputList$activatedSums[[synapseIndex-1]][trainEx, i]
      }
      
      gradBiasList[[synapseIndex]] = matrix(delta, 
                                            nrow=nrow(biasList[[synapseIndex]]), 
                                            ncol=ncol(biasList[[synapseIndex]]), byrow = T)
      
      while(synapseIndex > 1) {
        synapseIndex = synapseIndex - 1
        
        for(i in 1:nrow(gradWeightList[[synapseIndex]])) {
          for(j in 1:ncol(gradWeightList[[synapseIndex]])) {
            delta = reluDerivative(outputList$activatedSums[[synapseIndex]][trainEx, j]) * 
              sum(c(weightList[[synapseIndex+1]][j,]) * c(deltaWeightList[[synapseIndex+1]][j,]))
            deltaWeightList[[synapseIndex]][i,j] = delta
            
            if(synapseIndex == 1) {
              gradWeightList[[synapseIndex]][i,j] = delta * inputMat[trainEx,i]
              
            } else {
              gradWeightList[[synapseIndex]][i,j] = delta * outputList$activatedSums[[synapseIndex-1]][trainEx, i]
            }
          }
        }
        
        for(i in 1:nrow(gradBiasList[[synapseIndex]])) {
          for(j in 1:ncol(gradBiasList[[synapseIndex]])) {
            delta = reluDerivative(outputList$activatedSums[[synapseIndex]][trainEx, j]) * 
              sum(c(weightList[[synapseIndex+1]][j,]) * c(deltaWeightList[[synapseIndex+1]][j,]))
            gradBiasList[[synapseIndex]][i, j] = delta
          }
        }
        
      }
      
      synapseIndex = length(weightList)
      
      for(i in 1:synapseIndex) {
        weightList[[i]] = weightList[[i]] - (learningRate * gradWeightList[[i]])
        biasList[[i]] = biasList[[i]] - (learningRate * gradBiasList[[i]])
        #print(biasList[i])
      }
      
    }
    
    print(epochNum)
    
    if(epochNum%%10==0 || T)  {
      newOutput = forwardProp(origInput_mat, weightList, biasList)
      #print(round(newOutput$output))
      newBiasList = list()
      for(i in 1:length(biasList)) {
        newBiasList[[i]] = biasList[[i]][1,]
      }
      accuracy = calcAccuracy (round(newOutput$output), origOutput_mat)
      accuracy_test = test(input_test, output_test, weightList, newBiasList)
      currCost = CrossEntropyCost(origOutput_mat, newOutput$output)
      cat(epochNum, " ", "train: ", as.numeric(accuracy), ", test:  ", as.numeric(accuracy_test), " ", currCost)
      if(abs(currCost - prevCost) < .0001 && F) {
        break
      } else {
        prevCost = currCost
      }
    }
    
    epochNum = epochNum + 1
    randomSwap = sample(1:nrow(inputMat), nrow(inputMat), replace = F)
    
    inputMat = inputMat[randomSwap,]
    targetOutput = targetOutput[randomSwap,]
  }
  
  print("yes")
  newOutput = forwardProp(origInput_mat, weightList, biasList)
  print(round(newOutput$output), digits = 3)
  
  newBiasList = list()
  for(i in 1:length(biasList)) {
    newBiasList[[i]] = biasList[[i]][1,]
  }
  
  return (list("weights" = weightList, "biases" = newBiasList))
  
}

test = function (input_mat, output_mat, weightList, biasList) {
  for(i in 1:length(biasList)) {
    biasList[[i]] = matrix(rep(biasList[[i]], nrow(input_mat)), ncol = length(biasList[[i]]), byrow = T)
  }
  output = forwardProp(input_mat, weightList, biasList)
  return(calcAccuracy(round(output$output), output_mat))
}

library(googlesheets)
gs_sheet = gs_title("FRC 2019 Match Scouting")
scout_sheet = gs_read(gs_sheet)

data_input = data.frame(alliance = integer(),
                        crossed_line = double(),
                        hatches_auto_center = double(), hatches_auto_lv1 = double(), hatches_auto_lv2 = double(), hatches_auto_lv3 = double(),
                        balls_auto_center = double(), balls_auto_lv1 = double(), balls_auto_lv2 = double(), balls_auto_lv3 = double(),
                        pre_telop_move = integer(),
                        hatches_teleop_center = double(), hatches_teleop_lv1 = double(), hatches_teleop_lv2 = double(), hatches_teleop_lv3 = double(),
                        balls_teleop_center = double(), balls_teleop_lv1 = double(), balls_teleop_lv2 = double(), balls_teleop_lv3 = double(),
                        climb_level = double(),
                        disconnect = integer(), stringsAsFactors = F)
data_output = data.frame(winner = integer())

data_input_col = c(1, 3:12, 14:23)
index_input = 1
index_output = 1
for(i in 3:64) {
  if(is.na(scout_sheet[i, 1])) {
    if(i %% 7 == 2) {
      next
    } else if(scout_sheet[i-1, 1] == 0) {
      scout_sheet[i, 1] = 0
    } else if (scout_sheet[i-1, 1] == 1) {
      scout_sheet[i, 1] = 1
    } 
  } else if(scout_sheet[i, 1] == "Blue") {
    scout_sheet[i, 1] = 0
  } else if (scout_sheet[i, 1] == "Red") {
    scout_sheet[i, 1] = 1
  }
 
  data_input[index_input,] = scout_sheet[i, data_input_col]
  
  if(!is.na(scout_sheet[i, ncol(scout_sheet)])) {
    data_output[index_output,] = scout_sheet[i, ncol(scout_sheet)]
    index_output = index_output + 1
  }
  
  index_input = index_input + 1
}

data_input = data.matrix(data_input)
data_output = data.matrix(data_output)

max_hatches_center = 16
max_hatches_lv1 = 8
max_hatches_lv2 = 8
max_hatches_lv3 = 8

# normalizing "cross line"
data_input[,2] = data_input[,2] *.5 
# normalizing hatches
data_input[,c(3,12)] = ifelse(data_input[,c(3,12)] > max_hatches_center, 1, data_input[,c(3,12)]/ max_hatches_center)
data_input[,c(4,13)] = ifelse(data_input[,c(4,13)] > max_hatches_lv1, 1, data_input[,c(4,13)] / max_hatches_lv1 )
data_input[,c(5,14)] = ifelse(data_input[,c(5,14)] > max_hatches_lv2, 1, data_input[,c(5,14)] / max_hatches_lv2 )
data_input[,c(6,15)] = ifelse(data_input[,c(6,15)] > max_hatches_lv3, 1, data_input[,c(6,15)] / max_hatches_lv3 )

max_balls_center = 16
max_balls_lv1 = 8
max_balls_lv2 = 8
max_balls_lv3 = 8

data_input[,c(7,16)] = ifelse(data_input[,c(7,16)] > max_balls_center, 1, data_input[,c(7,16)]/ max_balls_center)
data_input[,c(8,17)] = ifelse(data_input[,c(8,17)] > max_balls_lv1, 1, data_input[,c(8,17)] / max_balls_lv1 )
data_input[,c(9,18)] = ifelse(data_input[,c(9,18)] > max_balls_lv2, 1, data_input[,c(9,18)] / max_balls_lv2 )
data_input[,c(10,19)] = ifelse(data_input[,c(10,19)] > max_balls_lv3, 1, data_input[,c(10,19)] / max_balls_lv3 )

data_input[,20] = data_input[,20] * .5 # normalizing "climb level"

input = matrix(nrow = nrow(data_input)/6, ncol = ncol(data_input)*6)
dim(input)
index = 1
index_data_first = 1
index_data_second = ncol(data_input)
for(i in 1:nrow(data_input)) {
  if(i %% 6 == 1 && i != 1) {
    index = index + 1
    index_data_first = 1
    index_data_second = ncol(data_input)
  }
 
  input[index, c(index_data_first : index_data_second)] = data_input[i,]

  index_data_first = index_data_first + ncol(data_input)
  index_data_second = index_data_second + ncol(data_input)
}

train_index = sample(1:nrow(input), round(.75 * nrow(input)))

input_train = input[train_index,]

output = matrix(data_output[train_index,])
output_train = matrix(nrow = nrow(input_train), ncol = 2)
for(i in 1:nrow(output)) {
  if(output[i,1] == 0) {
    output_train[i,] = c(1,0)
  } else {
    output_train[i,] = c(0,1)
  }
}

input_test = input[-train_index,]

output = matrix(data_output[-train_index,])
output_test = matrix(nrow = nrow(input_test), ncol = 2)
for(i in 1:nrow(output)) {
  if(output[i,1] == 0) {
    output_test[i,] = c(1,0)
  } else {
    output_test[i,] = c(0,1)
  }
}

input_train
output_train
input_test
output_test

numTrainingExamples = nrow(input_train)
numLayers = 3
eluAlpha = .7
learningRate = .01
epoch = 100
topology = c(ncol(input_train),74,2)

weightList = initWeightMats(topology)
biasList = initBiasMats(topology, numTrainingExamples)
outputList = forwardProp(input_train, weightList, biasList)
outputList$output

parameters = SGD(input_train, weightList, biasList, outputList, output_train, learningRate, epoch, input_test, output_test)

test(input_test, output_test, parameters$weights, parameters$biases)





