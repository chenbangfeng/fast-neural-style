require 'nngraph'
require 'InstanceNormalization'

local function shortcut()
  return nn.Sequential():add(nn.Narrow(4,3,-3)):add(nn.Narrow(3,3,-3))
end

-- Residual block for the network.
local function resblock(n,nInputPlane,inputSize, instance)
  local block = nn.Sequential()
  local s = nn.Sequential()
  s:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  s:add(nn.SpatialConvolution(nInputPlane,n,3,3,1,1))
  --s:add(nn.SpatialConvolution(nInputPlane,n,3,3,1,1))
  if instance then
    s:add(nn.InstanceNormalization(n))
  else
    s:add(nn.SpatialBatchNormalization(n))
  end
  
  s:add(nn.ReLU(true))
  s:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  s:add(nn.SpatialConvolution(n,n,3,3,1,1))
  --s:add(nn.SpatialConvolution(n,n,3,3,1,1))
  if instance then
    s:add(nn.InstanceNormalization(n))
  else
    s:add(nn.SpatialBatchNormalization(n))
  end

    block:add(nn.ConcatTable()
    :add(s)
    :add(shortcut()))
  block:add(nn.CAddTable(true))
  return block
end

local function build_conv_block(dim, use_instance_norm)
  local conv_block = nn.Sequential()
  local p = 0
  conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  if use_instance_norm == 1 then
    conv_block:add(nn.InstanceNormalization(dim))
  else
    conv_block:add(nn.SpatialBatchNormalization(dim))
  end
  conv_block:add(nn.ReLU(true))
  conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  if use_instance_norm == 1 then
    conv_block:add(nn.InstanceNormalization(dim))
  else
    conv_block:add(nn.SpatialBatchNormalization(dim))
  end
  return conv_block
end


local function build_res_block(dim, use_instance_norm)
  local conv_block = build_conv_block(dim, use_instance_norm)
  local res_block = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(conv_block)
  concat:add(nn.Identity())
  res_block:add(concat):add(nn.CAddTable())
  return res_block
end


function transferNet(instance)
  local model = nn.Sequential()
  --model:add(nn.SpatialReflectionPadding(40,40))
  model:add(nn.SpatialReflectionPadding(4, 4, 4, 4))
  model:add(nn.SpatialConvolution(3,32,9,9,1,1,0,0))
  if instance then
    model:add(nn.InstanceNormalization(32))
  else
    model:add(nn.SpatialBatchNormalization(32))
  end
  
  model:add(nn.ReLU(true))
  model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolution(32,64,3,3,2,2,0,0))
  --model:add(nn.SpatialConvolution(32,64,3,3,2,2,1,1))
  if instance then
    model:add(nn.InstanceNormalization(64))
  else
    model:add(nn.SpatialBatchNormalization(64))
  end
  
  model:add(nn.ReLU(true))
  model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolution(64,128,3,3,2,2,0,0))
  --model:add(nn.SpatialConvolution(64,128,3,3,2,2,1,1))
  if instance then
    model:add(nn.InstanceNormalization(128))
  else
    model:add(nn.SpatialBatchNormalization(128))
  end
  
  model:add(nn.ReLU(true))
  for i=1,5 do
      model:add(build_res_block(128,instance))
  end
  --model:add(nn.SpatialFullConvolution(128,64,3,3,2,2,1,1,1,1))
  model:add(nn.SpatialUpSamplingNearest(2))
  model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolution(128,64,3,3,1,1,0,0))
  if instance then
    model:add(nn.InstanceNormalization(64))
  else
    model:add(nn.SpatialBatchNormalization(64))
  end

  model:add(nn.ReLU(true))
  --model:add(nn.SpatialFullConvolution(64,32,3,3,2,2,1,1,1,1))
  model:add(nn.SpatialUpSamplingNearest(2))
  model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolution(64,32,3,3,1,1,0,0))
  if instance then
    model:add(nn.InstanceNormalization(32))
  else
    model:add(nn.SpatialBatchNormalization(32))
  end
  
  model:add(nn.ReLU(true))
  --model:add(nn.SpatialFullConvolution(32,3,9,9,1,1,4,4))
  model:add(nn.SpatialReflectionPadding(4, 4, 4, 4))
  model:add(nn.SpatialConvolution(32,3,9,9,1,1,0,0))
  if instance then
    model:add(nn.InstanceNormalization(3))
  else 
    model:add(nn.SpatialBatchNormalization(3))
  end
  
  --model:add(nn.Sigmoid())
  model:add(nn.Tanh())  
  model:add(nn.AddConstant(1))
  model:add(nn.MulConstant(127.5))
  return model
end


local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = nil
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.updateloss = false
end

function ContentLoss:setTarget(target)
  self.target = target
end

function ContentLoss:enableUpdateLoss(updateloss)
  self.updateloss = updateloss
end

function ContentLoss:updateOutput(input)
  if(self.updateloss and (nil ~= self.target)) then
    if input:nElement() == self.target:nElement() then
      self.loss = self.crit:forward(input, self.target) * self.strength
    else
      print('WARNING: Skipping content loss')
    end
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = target
  self.loss = 0
  
  self.gram = GramMatrix()
  self.G = nil
  self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target)
  dG:div(input:nElement())
  self.gradInput = self.gram:backward(input, dG)
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local N, C = input:size(1), input:size(2)
  local H, W = input:size(3), input:size(4)
  self.x_diff:resize(N, 3, H - 1, W - 1)
  self.y_diff:resize(N, 3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end
