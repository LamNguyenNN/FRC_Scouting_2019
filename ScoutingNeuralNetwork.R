initWeightMats = function(topo) {
  i = 1
  weightMats = list()
  while(i < length(topo)) {
    weightMats[[i]] = matrix(data = rnorm(n=topo[i]*topo[i+1], mean=0, sd=1), nrow=topo[i], ncol=topo[i+1], byrow=T)
    i = i + 1
  }
  return (weightMats)
}

loadWeights = function(weightsCSV) {
  
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

SGD = function(input_train, weightList, biasList, outputList, output_train, learningRate, epoch, input_test, output_test) {
  origInput_mat = input_train
  origOutput_mat = output_train
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
    for(trainEx in 1:nrow(output_train)) {
      outputList = forwardProp(input_train, weightList, biasList)
      if(anyNA(outputList$output)) {
        #print(outputList$activatedSums)
        #print(outputList$unact)
        #print(outputList$output)
        #print("SGD")
        print(input_train[trainEx,])
        print(epochNum)
        print(trainEx)
        stop()
      }
      
      delta = outputList$output[trainEx,] - output_train[trainEx,]
      
      deltaWeightList[[synapseIndex]] = matrix(delta, 
                                               nrow=nrow(weightList[[synapseIndex]]), 
                                               ncol=ncol(weightList[[synapseIndex]]), byrow = T)
      
      gradWeightList[[synapseIndex]][] = deltaWeightList[[synapseIndex]][] * outputList$activatedSums[[synapseIndex-1]][trainEx,]
      
      gradBiasList[[synapseIndex]] = matrix(delta, 
                                            nrow=nrow(biasList[[synapseIndex]]), 
                                            ncol=ncol(biasList[[synapseIndex]]), byrow = T)
      
      while(synapseIndex > 1) {
        synapseIndex = synapseIndex - 1
        delta = reluDerivative(outputList$activatedSums[[synapseIndex]][trainEx,]) * 
          rowSums(weightList[[synapseIndex+1]] * deltaWeightList[[synapseIndex+1]])
        
        for(i in 1:nrow(deltaWeightList[[synapseIndex]])) {
          deltaWeightList[[synapseIndex]][i,] = delta
        }
        
        for(i in 1:nrow(gradBiasList[[synapseIndex]])) {
          gradBiasList[[synapseIndex]][i,] = delta
        }
        
        if(synapseIndex == 1) {
          gradWeightList[[synapseIndex]] = deltaWeightList[[synapseIndex]] * input_train[trainEx,]
        } else {
          gradWeightList[[synapseIndex]] = deltaWeightList[[synapseIndex]] * outputList$activatedSums[[synapseIndex-1]][trainEx, ]
        }
        
      }
      
      synapseIndex = length(weightList)
      
      for(i in 1:synapseIndex) {
        weightList[[i]] = weightList[[i]] - (learningRate * gradWeightList[[i]])
        biasList[[i]] = biasList[[i]] - (learningRate * gradBiasList[[i]])
      }
      
    }
    
    if(epochNum%%5==0 || T)  {
      newOutput = forwardProp(origInput_mat, weightList, biasList)
      newBiasList = list()
      for(i in 1:length(biasList)) {
        newBiasList[[i]] = biasList[[i]][1,]
      }
      accuracy = calcAccuracy (round(newOutput$output), origOutput_mat)
      accuracy_test = test(input_test, output_test, weightList, newBiasList)
      currCost = CrossEntropyCost(origOutput_mat, newOutput$output)
      cat("epoch: ", epochNum, ", ", "train acc: ", as.numeric(accuracy), ", test acc:  ", 
          as.numeric(accuracy_test), ", loss:", currCost, "\n")
      if(abs(currCost - prevCost) < .0001 && accuracy > .95) {
        break
      } else {
        prevCost = currCost
      }
    }
    
    epochNum = epochNum + 1
    randomSwap = sample(1:nrow(input_train), nrow(input_train), replace = F)
    
    input_train = input_train[randomSwap,]
    output_train = output_train[randomSwap,]
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

loadSheet = function(sheet = "FRC 2019 Match Scouting") {
  library(googlesheets)
  gs_sheet = gs_title(sheet)
  scout_sheet = gs_read(gs_sheet)
  
  data = data.frame(alliance = integer(),
                    hatches_setup = integer(), balls_setup = integer(),
                    cross_line = double(),
                    hatches_auto_center = double(), hatches_auto_lv1 = double(), hatches_auto_lv2 = double(), hatches_auto_lv3 = double(),
                    balls_auto_center = double(), balls_auto_lv1 = double(), balls_auto_lv2 = double(), balls_auto_lv3 = double(),
                    prep_teleop = integer(),
                    hatches_teleop_center = double(), hatches_teleop_lv1 = double(), hatches_teleop_lv2 = double(), hatches_teleop_lv3 = double(),
                    balls_teleop_center = double(), balls_teleop_lv1 = double(), balls_teleop_lv2 = double(), balls_teleop_lv3 = double(),
                    climb_level = double(),
                    radio_problems = integer(),
                    winner = integer(), stringsAsFactors = F)
  scout_sheet
  useful_columns = c(1, 3:25)
  index_input = 1
  for(i in 4:nrow(scout_sheet)) {
    if(is.na(scout_sheet[i, 1])) {
      data[index_input,] = scout_sheet[i, useful_columns]
      index_input = index_input + 1
      next
    } else if(scout_sheet[i, 1] == "Blue") {
      scout_sheet[i, 1] = 0
    } else if (scout_sheet[i, 1] == "Red") {
      scout_sheet[i, 1] = 1
    } else {
      next
    }
    
    data[index_input,] = scout_sheet[i, useful_columns]
    index_input = index_input + 1
  }
  
  data = data.matrix(data)
  data
  #Fixing "NA" in alliances, hatches_setup, balls_setup, output
  for(i in 1:nrow(data)) {
    if(is.na(data[i, "alliance"])) {
      if(i %% 6 == 2 || i %% 6 == 3) {
        data[i, "alliance"] = 1
      } else if (i %% 6 == 5 || i %% 6 == 0) {
        data[i, "alliance"] = 0
      }
    }
    
    if(is.na(data[i, "hatches_setup"]) && is.na(data[i, "balls_setup"])) {
      if(i %% 6 == 2 || i %% 6 == 5) {
        data[i, "hatches_setup"] = data[i-1, "hatches_setup"]
        data[i, "balls_setup"] = data[i-1, "balls_setup"]
      } else if(i %% 6 == 3 || i %% 6 == 0) {
        data[i, "hatches_setup"] = data[i-2, "hatches_setup"]
        data[i, "balls_setup"] = data[i-2, "balls_setup"]
      } 
    }
    
    if(is.na(data[i, "winner"])) {
      data[i, "winner"] = data[i-1, "winner"]
    }
    
  }
  data
  # normalizing "cross line"
  data[,"cross_line"] = data[,"cross_line"] *.5 
  
  # normalizing hatches
  max_hatches_center = 8
  max_hatches_rocket_level = 8
  
  data[,c("hatches_setup", "hatches_auto_center", "hatches_teleop_center")] = ifelse(
    data[,c("hatches_setup", "hatches_auto_center", "hatches_teleop_center")] > max_hatches_center, 
    1, data[,c("hatches_setup", "hatches_auto_center", "hatches_teleop_center")]/ max_hatches_center)
  
  data[,c("hatches_auto_lv1", "hatches_auto_lv2", "hatches_auto_lv3", "hatches_teleop_lv1", "hatches_teleop_lv2", "hatches_teleop_lv3")] =
    ifelse(data[,c("hatches_auto_lv1", "hatches_auto_lv2", "hatches_auto_lv3", 
                   "hatches_teleop_lv1", "hatches_teleop_lv2", "hatches_teleop_lv3")] > max_hatches_rocket_level,
           1, data[,c("hatches_auto_lv1", "hatches_auto_lv2", "hatches_auto_lv3", 
                      "hatches_teleop_lv1", "hatches_teleop_lv2", "hatches_teleop_lv3")] / max_hatches_rocket_level )
  
  max_balls_center = 8
  max_balls_rocket_level = 8
  
  data[,c("balls_setup", "balls_auto_center", "balls_teleop_center")] = ifelse(
    data[,c("balls_setup", "balls_auto_center", "balls_teleop_center")] > max_balls_center, 
    1, data[,c("balls_setup", "balls_auto_center", "balls_teleop_center")]/ max_balls_center)
  
  data[,c("balls_auto_lv1", "balls_auto_lv2", "balls_auto_lv3", "balls_teleop_lv1", "balls_teleop_lv2", "balls_teleop_lv3")] =
    ifelse(data[,c("balls_auto_lv1", "balls_auto_lv2", "balls_auto_lv3", 
                   "balls_teleop_lv1", "balls_teleop_lv2", "balls_teleop_lv3")] > max_balls_rocket_level,
           1, data[,c("balls_auto_lv1", "balls_auto_lv2", "balls_auto_lv3", 
                      "balls_teleop_lv1", "balls_teleop_lv2", "balls_teleop_lv3")] / max_balls_rocket_level )
  
  #normalizing "climb level"
  data[,"climb_level"] = data[,"climb_level"] / 3 
  data
  
  library(gtools)
  permute_matrix = permutations(3, 3)
  
  data_permute = matrix(nrow = nrow(data)*36, ncol = ncol(data))
  
  permute_index1 = 1
  permute_index2 = 1
  index1 = 1
  index2 = 3
  permute_counter1 = 0
  permute_counter2 = 3
  while(index1 <= nrow(data_permute)) {
    while(permute_index1 <= nrow(permute_matrix)) {
      while(permute_index2 <= nrow(permute_matrix)) {
        data_permute[index1:index2,] = data[permute_matrix[permute_index1,] + permute_counter1,] #1 2 3, 7 8 9
        index1 = index1 + 3
        index2 = index2 + 3
        data_permute[index1:index2,] = data[permute_matrix[permute_index2,] + permute_counter2,] #4 5 6, 10 11 12
        permute_index2 = permute_index2 + 1
        index1 = index1 + 3
        index2 = index2 + 3
      }
      permute_index1 = permute_index1 + 1
      permute_index2 = 1
    }
    permute_counter1 = permute_counter1 + 6
    perrmute_counter2 = permute_counter2 + 6
    permute_index1 = 1
  }
  data_permute
  data_input = matrix(nrow = nrow(data_permute) / 6, ncol = 3 + ((ncol(data_permute)-4)*6) )
  data_output = matrix(nrow = nrow(data_permute) / 6, ncol = 3)
  
  index_data = 1
  index1 = 1
  index2 = 1
  index_permute = 1
  while(index_data <= nrow(data_input)) {
    if(index_permute %% 6 == 1) {
      index2 = index1 + ncol(data_permute) - 2
      data_input[index_data, index1:index2] = data_permute[index_permute, 1:(ncol(data_permute)-1)]
      index1 = index2 + 1
      index_permute = index_permute + 1
    } else {
      index2 = index1 + ncol(data_permute) - 5
      data_input[index_data, index1:index2] = data_permute[index_permute, 4:(ncol(data_permute)-1)]
      index1= index2 + 1
      index_permute = index_permute + 1
    }
    if(index1 > ncol(data_input)) {
      index1 = 1
      index_data = index_data + 1
    }
  }
  
  index = 1
  for(i in 1:nrow(data_permute)) {
    if(i %% 6 == 1) {
      if(data_permute[i, ncol(data_permute)] == 0) {
        data_output[index,] = c(1,0,0)
      } else if (data_permute[i, ncol(data_permute)] == 1) {
        data_output[index,] = c(0,1,0)
      } else if (data_permute[i, ncol(data_permute)] == 2) {
        data_output[index,] = c(0,0,1)
      }
      index = index + 1
    }
  }
  return (list("input" = data_input, "output" = data_output))
}

sheet = "FRC 2019 Match Scouting"
data = loadSheet(sheet)
data_input = data$input
data_output = data$output
train_index = sample(1:nrow(data_input), round(.75 * nrow(data_input)))

input_train = data_input[train_index,]
output_train = data_output[train_index,]

input_test = data_input[-train_index,]
output_test = data_output[-train_index,]
output_train
numTrainingExamples = nrow(input_train)
numTrainingExamples
numLayers = 3
eluAlpha = .7
learningRate = .005
epoch = 100
topology = c(ncol(input_train),74,3)

weightList = initWeightMats(topology)
biasList = initBiasMats(topology, numTrainingExamples)
outputList = forwardProp(input_train, weightList, biasList)
#making changes
parameters = SGD(input_train, weightList, biasList, outputList, output_train, learningRate, epoch, input_test, output_test)
library("rJava")
#test(input_test, output_test, parameters$weights, parameters$biases)

#for(i in 1:length(weightList)) {
#  write.table(weightList[[i]], file = paste("weights", i,".csv", sep = ""), row.names = F, col.names = F, sep = " ")
#  write.table(biasList[[i]], file = paste("biases", i,".csv", sep = ""), row.names = F, col.names = F, sep = " ")
#}
#w = read.csv(file = "weights1.csv", header = F, sep = " ")
#w = data.matrix(w)
#dim(w)
