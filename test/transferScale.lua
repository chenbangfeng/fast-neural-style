require 'nngraph'
require 'imgtools'
require 'cutorch'
require 'cunn'
require 'InstanceNormalization'

local cmd = torch.CmdLine()

cmd:option('-content_img','image_sets/tubingen.jpg','The image to apply style transfer on')
cmd:option('-transfer_model','models/newlytrainedmodel.t7','The model of transferring')
cmd:option('-output','output.jpg','The path to output')
cmd:option('-output_size',1024,'Maximum edge of the output image')

local saveIm = image.saveJPG
local loadIm = image.loadJPG

function main(params)
	timer = torch.Timer()
	transfer_model = torch.load(params.transfer_model):cuda()

	img = image.scale(loadIm(params.content_img), params.output_size, 'bilinear')	
	local image = toBRG(img)
	
	local content_batch_size = torch.LongStorage(4)
	content_batch_size[1] = 1
	for i=1,3 do
		content_batch_size[i+1] = (#image)[i]
	end
	image = torch.reshape(image,content_batch_size)
	image = image:cuda()
	local newimg = transfer_model:forward(image)
	newimg = BRGtoRGB(newimg:squeeze():double())
	newimg:div(255.0)
	saveIm(params.output, newimg)
	print('Transfer complete in '.. timer:time().real ..' seconds!')
end

local params = cmd:parse(arg)
main(params)
