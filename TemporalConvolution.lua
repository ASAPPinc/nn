local TemporalConvolution, parent = torch.class('nn.TemporalConvolution', 'nn.Module')

function TemporalConvolution:__init(inputFrameSize, outputFrameSize, kW, dW)
   parent.__init(self)

   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.outputFrameSize = outputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.bias = torch.Tensor(outputFrameSize)
   self.gradWeight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.gradBias = torch.Tensor(outputFrameSize)

   self:reset()
end

function TemporalConvolution:clearState()
   for _, buffer in ipairs({'Input', 'OutputSized', 'GradInput', 'SpatialWeightSized'}) do
      local bufferName = 'spatial' .. buffer
      if self[bufferName] then self[bufferName]:set() end
   end
   return parent.clearState(self)
end

function TemporalConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.inputFrameSize)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)   
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function TemporalConvolution:updateOutput(input)
    self.finput = self.finput or input.new()
    self.fgradInput = self.fgradInput or input.new()

    self.spatialInput = self.spatialInput or input.new()
    self.spatialOutputSized = self.spatialOutputSized or input.new()
    self.spatialWeightSized = self.spatialWeightSized or input.new()

    input.THNN.TemporalConvolution_updateOutput(
      input:cdata(), self.output:cdata(),
      self.weight:cdata(), self.bias:cdata(),
      self.finput:cdata(), self.fgradInput:cdata(),
      self.spatialInput:cdata(), self.spatialOutputSized:cdata(),
      self.spatialWeightSized:cdata(),
      self.kW, self.dW,
      self.inputFrameSize, self.outputFrameSize
    )
   return self.output
end

function TemporalConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.spatialInput = self.spatialInput or input.new()
      self.spatialOutputSized = self.spatialOutputSized or input.new()
      self.spatialGradInput = self.spatialGradInput or input.new()
      self.spatialWeightSized = self.spatialWeightSized or input.new()
      input.THNN.TemporalConvolution_updateGradInput(
        input:cdata(), gradOutput:cdata(),
        self.gradInput:cdata(), self.weight:cdata(),
        self.spatialInput:cdata(), self.spatialOutputSized:cdata(), self.spatialGradInput:cdata(),
        self.spatialWeightSized:cdata(),
        self.kW, self.dW
      )
      return self.gradInput
   end
end

function TemporalConvolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.spatialInput = self.spatialInput or input.new()
   self.spatialOutputSized = self.spatialOutputSized or input.new()
   self.spatialWeightSized = self.spatialWeightSized or input.new()
   input.THNN.TemporalConvolution_accGradParameters(
     input:cdata(), gradOutput:cdata(),
     self.gradWeight:cdata(), self.gradBias:cdata(),
     self.spatialInput:cdata(), self.spatialOutputSized:cdata(),
     self.spatialWeightSized:cdata(),
     self.kW, self.dW, scale
   )
end

-- we do not need to accumulate parameters when sharing
TemporalConvolution.sharedAccUpdateGradParameters = TemporalConvolution.accUpdateGradParameters
