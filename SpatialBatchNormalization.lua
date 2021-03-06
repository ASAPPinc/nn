--[[
   This file implements Batch Normalization as described in the paper:
   "Batch Normalization: Accelerating Deep Network Training
                         by Reducing Internal Covariate Shift"
                by Sergey Ioffe, Christian Szegedy

   This implementation is useful for inputs coming from convolution layers.
   For Non-convolutional layers, see BatchNormalization.lua

   The operation implemented is:
   y =     ( x - mean(x) )
   -------------------- * gamma + beta
   standard-deviation(x)
   where gamma and beta are learnable parameters.

   The learning of gamma and beta is optional.

   Usage:
   with    learnable parameters: nn.SpatialBatchNormalization(N [,eps] [,momentum])
                                 where N = dimensionality of input
   without learnable parameters: nn.SpatialBatchNormalization(N [,eps] [,momentum], false)

   eps is a small value added to the variance to avoid divide-by-zero.
       Defaults to 1e-5

   In training time, this layer keeps a running estimate of it's computed mean and std.
   The running sum is kept with a default momentum of 0.1 (unless over-ridden)
   In test time, this running mean/std is used to normalize.

]]--
local BN,parent = torch.class('nn.SpatialBatchNormalization', 'nn.Module')
local THNN = require 'nn.THNN'

BN.__version = 2

function BN:__init(nFeature, eps, momentum, affine)
   parent.__init(self)
   assert(nFeature and type(nFeature) == 'number',
          'Missing argument #1: Number of feature planes. ')
   assert(nFeature ~= 0, 'To set affine=false call SpatialBatchNormalization'
     .. '(nFeature,  eps, momentum, false) ')
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = true
   end
   self.eps = eps or 1e-5
   self.train = true
   self.momentum = momentum or 0.1

   self.running_mean = torch.zeros(nFeature)
   self.running_var = torch.ones(nFeature)
   if self.affine then
      self.weight = torch.Tensor(nFeature)
      self.bias = torch.Tensor(nFeature)
      self.gradWeight = torch.Tensor(nFeature)
      self.gradBias = torch.Tensor(nFeature)
      self:reset()
   end
end

function BN:reset()
   if self.weight then
      self.weight:uniform()
   end
   if self.bias then
      self.bias:zero()
   end
   self.running_mean:zero()
   self.running_var:fill(1)
end

function BN:updateOutput(input)
   assert(input:dim() == 4, 'only mini-batch supported (4D tensor), got '
             .. input:dim() .. 'D tensor instead')

   self.output:resizeAs(input)
   self.save_mean = self.save_mean or input.new()
   self.save_mean:resizeAs(self.running_mean)
   self.save_std = self.save_std or input.new()
   self.save_std:resizeAs(self.running_var)

   input.THNN.SpatialBatchNormalization_updateOutput(
      input:cdata(),
      self.output:cdata(),
      THNN.optionalTensor(self.weight),
      THNN.optionalTensor(self.bias),
      self.running_mean:cdata(),
      self.running_var:cdata(),
      self.save_mean:cdata(),
      self.save_std:cdata(),
      self.train,
      self.momentum,
      self.eps)

   return self.output
end

local function backward(self, input, gradOutput, scale, gradInput, gradWeight, gradBias)
   assert(input:dim() == 4, 'only mini-batch supported')
   assert(gradOutput:dim() == 4, 'only mini-batch supported')
   assert(self.train == true, 'should be in training mode when self.train is true')
   assert(self.save_mean and self.save_std, 'must call :updateOutput() first')

   scale = scale or 1
   if gradInput then
      gradInput:resizeAs(gradOutput)
   end

   input.THNN.SpatialBatchNormalization_backward(
      input:cdata(),
      gradOutput:cdata(),
      THNN.optionalTensor(gradInput),
      THNN.optionalTensor(gradWeight),
      THNN.optionalTensor(gradBias),
      THNN.optionalTensor(self.weight),
      self.save_mean:cdata(),
      self.save_std:cdata(),
      scale)

   return self.gradInput
end

function BN:backward(input, gradOutput, scale)
   return backward(self, input, gradOutput, scale, self.gradInput, self.gradWeight, self.gradBias)
end

function BN:updateGradInput(input, gradOutput)
   return backward(self, input, gradOutput, 1, self.gradInput)
end

function BN:accGradParameters(input, gradOutput, scale)
   return backward(self, input, gradOutput, scale, nil, self.gradWeight, self.gradBias)
end

function BN:read(file, version)
   parent.read(self, file)
   if version < 2 then
      if self.running_std then
         self.running_var = self.running_std:pow(-2):add(-self.eps)
         self.running_std = nil
      end
   end
end

function BN:clearState()
   -- first 5 buffers are not present in the current implementation,
   -- but we keep them for cleaning old saved models
   nn.utils.clear(self, {
      'buffer',
      'buffer2',
      'centered',
      'std',
      'normalized',
      'save_mean',
      'save_std',
   })
   return parent.clearState(self)
end
