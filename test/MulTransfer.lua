require 'lfs'
require 'os'
require 'nngraph'
require 'imgtools'
require 'cutorch'
require 'cunn'
require 'InstanceNormalization'
local utils = require 'utils'

local cmd = torch.CmdLine()
cmd:option('-model_path', 'transfer_models', 'The path of model of transferring')
cmd:option('-content_path', 'content_images', 'The path of image to apply style transfer on')
cmd:option('-output_size',800,'Maximum edge of the output image')
cmd:option('-output_path','output_images','The path to output')
cmd:option('-gpu', 0)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)
cmd:option('-cudnn_benchmark', 0)
local saveIm = image.saveJPG
local loadIm = image.loadJPG
local function main(params)
	transfer_model_files = {}
	model_file_num = 1
	for file in lfs.dir(params.model_path) do
        if file ~= "." and file ~= ".." then
            local f = params.model_path..'/'..file
            local attr = lfs.attributes (f)
            assert (type(attr) == "table")
            if attr.mode ~= "directory" then
				transfer_model_files[model_file_num] = f
				model_file_num = model_file_num + 1
			end
		end
	end
	
	content_img_files = {}
	img_file_num = 1
	for file in lfs.dir(params.content_path) do
        if file ~= "." and file ~= ".." then
            local f = params.content_path..'/'..file
            local attr = lfs.attributes (f)
            assert (type(attr) == "table")
            if attr.mode ~= "directory" then
				content_img_files[img_file_num] = f
				img_file_num = img_file_num + 1
			end
		end
	end

	local dtype, use_cudnn = utils.setup_gpu(params.gpu, params.backend, params.use_cudnn == 1)
	

	for i=1,#transfer_model_files do
	    for j=1,#content_img_files do
		    print("transfer content<"  .. content_img_files[j] .. "> model<"  .. transfer_model_files[i] .. ">")
			timer = torch.Timer()
			transfer_model = torch.load(transfer_model_files[i])
  			transfer_model:type(dtype)
    		--transfer_model:evaluate()	
			if use_cudnn then
		    	cudnn.convert(transfer_model, cudnn)
		    	if params.cudnn_benchmark == 0 then
		      		cudnn.benchmark = false
		      		cudnn.fastest = true
		    	end
		  	end

			img = image.scale(loadIm(content_img_files[j]), params.output_size, 'bilinear')	
			local image = toBRG(img)
			
			local content_batch_size = torch.LongStorage(4)
			content_batch_size[1] = 1
			for k=1,3 do
				content_batch_size[k+1] = (#image)[k]
			end
			image = torch.reshape(image,content_batch_size)
			image = image:type(dtype)
			local newimg = transfer_model:forward(image)
			newimg = BRGtoRGB(newimg:squeeze():double())
			newimg:div(255.0)
			local strmodelname = string.sub(transfer_model_files[i], 17, string.len(transfer_model_files[i])-3)
			local strcontentname = string.sub(content_img_files[j], 16, string.len(content_img_files[j])-4)
			saveIm(params.output_path..'/'..strmodelname.."_"..strcontentname..'.jpg', newimg)
			print('Transfer complete in '.. timer:time().real ..' seconds!')
	    end
	end

end

local params = cmd:parse(arg)
main(params)
