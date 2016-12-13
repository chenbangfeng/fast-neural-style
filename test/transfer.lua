require 'nngraph'
require 'imgtools'
require 'cutorch'
require 'cunn'
require 'InstanceNormalization'
local utils = require 'utils'

local cmd = torch.CmdLine()

cmd:option('-content_img','image_sets/tubingen.jpg','The image to apply style transfer on')
cmd:option('-transfer_model','models/newlytrainedmodel.t7','The model of transferring')
cmd:option('-output','output.jpg','The path to output')
cmd:option('-output_size',false,'Maximum edge of the output image')
cmd:option('-gpu', 0)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)
cmd:option('-cudnn_benchmark', 0)

local saveIm = image.saveJPG
local loadIm = image.loadJPG

function main(params)
	local dtype, use_cudnn = utils.setup_gpu(params.gpu, params.backend, params.use_cudnn == 1)
	timer = torch.Timer()
	transfer_model = torch.load(params.transfer_model)
  	transfer_model:type(dtype)
        --transfer_model:evaluate()	
	if use_cudnn then
    	cudnn.convert(transfer_model, cudnn)
    	if params.cudnn_benchmark == 0 then
      		cudnn.benchmark = false
      		cudnn.fastest = true
    	end
  	end

	local image = toBRG(scale_pp(loadIm(params.content_img),params.output_size))
	local content_batch_size = torch.LongStorage(4)
	content_batch_size[1] = 1
	for i=1,3 do
		content_batch_size[i+1] = (#image)[i]
	end
	image = torch.reshape(image,content_batch_size)
	image = image:type(dtype)
	local newimg = transfer_model:forward(image)
	newimg = BRGtoRGB(newimg:squeeze():double())
	newimg:div(255.0)
	saveIm(params.output, newimg)
	print('Transfer complete in '.. timer:time().real ..' seconds!')
end

local params = cmd:parse(arg)
main(params)
