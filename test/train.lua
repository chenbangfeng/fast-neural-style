require 'networks'
require 'imgtools'
require 'optim'
require 'cutorch'
require 'cunn'
require 'DeepDreamLoss'
coco = require 'coco'
require 'loadcaffe'
local utils = require 'utils'
local cmd = torch.CmdLine()

cmd:option('-style_img', 'style_images/starry.jpg', 'Style image to train on the image set')
-- next option is going to be the training set option when I get around
cmd:option('-training_img', 'image_sets/tubingen.jpg', 'Placeholder single training image')
cmd:option('-modelfile', 'models/VGG_ILSVRC_16_layers.caffemodel', 'Model used for perceptual losses')
cmd:option('-protofile', 'models/VGG_ILSVRC_16_layers_deploy.prototxt', 'prototxt of the perception model')
cmd:option('-style_weight', 1e1)
cmd:option('-content_weight', 1.0)
cmd:option('-tv_weight', 1e-6)
cmd:option('-style_layers', 'relu1_2,relu2_2,relu3_3,relu4_3')
cmd:option('-content_layers', 'relu2_2')
cmd:option('-deepdream_layers', '')
cmd:option('-iterations', 100, 'Number of iterations to run for training')
cmd:option('-trained_model', 'models/newlytrainedmodel.t7', '')
cmd:option('-trained_epoch', 2,'')
cmd:option('-training_batchsize', 2,'batch size (default value is 2)')
cmd:option('-image_size', 256, 'Maximum height / width of generated image')
cmd:option('-learning_rate', 1e-3)
cmd:option('-test','test_image.jpg','Test Image path')
cmd:option('-normalize_gradients', false)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-use_instance_norm', 1)
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda', 'cuda|opencl')
cmd:option('-dream_strength', 1e-5)
cmd:option('-dream_max_grad', 200.0)

local loadIm = image.loadJPG

function build_modelname(output_image, epoch, iterations)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = 'models'
  return string.format('%s/%s_%d_%d.%s',directory, basename, epoch, iterations, 't7')
end

local function saveImage(transformNet, iter, testIm,output_image)
	local out = transformNet:forward(testIm)
    local ext = paths.extname(output_image) 
    local basename = paths.basename(output_image, ext)	
	local file_name = paths.concat('./','test' .. basename ..tostring(iter)..'.'..'jpg')
	out = BRGtoRGB(out:squeeze():double())
	out:div(255.0)
	image.saveJPG(file_name,out)
	print ('Saving Test Image @ iter : '..tostring(iter),'File name : ' .. file_name)
end


function prepare_VGG16(params, style_images)
        
 	local content_layers = params.content_layers:split(",")
  	local style_layers = params.style_layers:split(",")
  	local deepdream_layers = params.deepdream_layers:split(",")

  	-- Set up the network, inserting style and content loss modules
  	local loadcaffe_backend = 'nn'
  	local cnn = loadcaffe.load(params.protofile, params.modelfile, loadcaffe_backend):float()
    cnn:cuda()

  	local content_module, style_module = {}, {}
  	local next_content_idx, next_style_idx, next_deepdream_idx = 1, 1, 1
  	local net = nn.Sequential()
  	if params.tv_weight > 0 then
    	local tv_mod = nn.TVLoss(params.tv_weight):float()
        tv_mod:cuda()
    	net:add(tv_mod)
  	end
  	for i = 1, #cnn do
	    if next_content_idx <= #content_layers or next_style_idx <= #style_layers  or next_deepdream_idx <= #deepdream_layers then
	      	local layer = cnn:get(i)
	      	local name = layer.name
	      	local layer_type = torch.type(layer)
	      	local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
	      	if is_pooling and params.pooling == 'avg' then
	        	assert(layer.padW == 0 and layer.padH == 0)
	        	local kW, kH = layer.kW, layer.kH
	        	local dW, dH = layer.dW, layer.dH
	        	local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
	            avg_pool_layer:cuda()
	          	
	        	local msg = 'Replacing max pooling at layer %d with average pooling'
	        	print(string.format(msg, i))
	        	net:add(avg_pool_layer)
	      	else
	        	net:add(layer)
	      	end
	      	if name == deepdream_layers[next_deepdream_idx] then
	        	print("Setting up deepdream layer", i, ":", layer.name)
	        	local loss_module = nn.DeepDreamLoss(params.dream_strength, params.dream_max_grad):float()
	            loss_module:cuda()
	        	net:add(loss_module)
	        	next_deepdream_idx = next_deepdream_idx + 1
	      	end
	      	if name == content_layers[next_content_idx] then
	        	print("Setting up content layer", i, ":", layer.name)
	        	local norm = params.normalize_gradients
	        	local loss_module = nn.ContentLoss(params.content_weight, norm):float()
	            loss_module:cuda()
	        	net:add(loss_module)
	        	table.insert(content_module, loss_module)
	        	next_content_idx = next_content_idx + 1
	      	end
	      	if name == style_layers[next_style_idx] then
	        	print("Setting up style layer  ", i, ":", layer.name)
	        	local gram = GramMatrix():float()
	            gram = gram:cuda()
	    
	          	local target_features = net:forward(style_images):clone()
	          	local target = gram:forward(target_features):clone()
	          	target:div(target_features:nElement())
	        	local norm = params.normalize_gradients
	        	local loss_module = nn.StyleLoss(params.style_weight, target, norm):float()
	            loss_module:cuda()
	          	
	        	net:add(loss_module)
	        	table.insert(style_module, loss_module)
	        	next_style_idx = next_style_idx + 1
	      	end
    	end
  	end

  	-- We don't need the base CNN anymore, so clean it up to save memory.
  	cnn = nil
  	for i=1,#net.modules do
    	local module = net.modules[i]
    	if torch.type(module) == 'nn.SpatialConvolutionMM' then
        	-- remove these, not used, but uses gpu memory
        	module.gradWeight = nil
        	module.gradBias = nil
    	end
  	end
  	collectgarbage()
  	return net,content_module,style_module
end

local function main(params)
	local dtype, use_cudnn = utils.setup_gpu(params.gpu, params.backend, params.use_cudnn == 1)

	local testIm = ppBRG(toBRG(scale_pp(loadIm(params.test),false)))
	testIm:resize(1,3,testIm:size(2),testIm:size(3))
	testIm = testIm:type(dtype)

	local batch_size = params.training_batchsize
	style_img_tmp = ppBRG(toBRG(scale_pp(loadIm(params.style_img), params.image_size)))
	style_batch_size = torch.LongStorage(4)
	style_batch_size[1] = batch_size
	for i=1,3 do
		style_batch_size[i+1] = (#style_img_tmp)[i]
	end
	style_img = torch.Tensor(style_batch_size)
	for i=1,batch_size do
		style_img[i] = style_img_tmp
	end
	style_img = style_img:type(dtype)
	
	-- Set up the networks
	transfer_model = transferNet(params.use_instance_norm):type(dtype)
	if use_cudnn then cudnn.convert(transfer_model, cudnn) end
	transfer_model:training()
  	print(transfer_model)
	perception_model, content_module, style_module = prepare_VGG16(params, style_img)
	if use_cudnn then cudnn.convert(perception_model, cudnn) end

	for _, mod in ipairs(content_module) do
		mod:enableUpdateLoss(true)
		mod:setTarget(nil)
	end


	print("**************************************************************")

	annTypes = { 'instances', 'captions', 'person_keypoints' }
	dataType, annType = 'train2014', annTypes[2]; -- specify dataType/annType
	annFile = '/home/zmia/data/mscoco/annotations/'..annType..'_'..dataType..'.json'
	cocoApi=coco.CocoApi(annFile)
	imgIds = cocoApi:getImgIds()
	local optParams, gradParams = transfer_model:getParameters()
	for nepoch=1,params.trained_epoch do
		local  num_iterations = 0 
		for imgId=1,imgIds:numel(),batch_size do
			transfer_model:zeroGradParameters()
			perception_model:zeroGradParameters()

			local content_img = torch.Tensor(batch_size,3,params.image_size,params.image_size)
			local content_img_trans = torch.Tensor(batch_size,3,params.image_size,params.image_size)
			local img = cocoApi:loadImgs(imgIds[imgId])[1]
			for i=1,batch_size do
				imag_file_name = '/home/zmia/data/mscoco/'..dataType..'/'..img.file_name
				content_img[i] = toBRG(scale_crop(loadIm(imag_file_name,3),params.image_size))
				content_img_trans[i] = content_img[i]:clone()
				content_img_trans[i] = ppBRG(content_img_trans[i])
				content_img[i]:mul(255.0)
			end

			content_img = content_img:type(dtype)
			content_img_trans = content_img_trans:type(dtype)


			local y = perception_model:forward(content_img_trans)
			for _, mod in ipairs(content_module) do
				mod:setTarget(mod.output:clone())
			end

			local dy = content_img_trans.new(#y):zero()

			y = nil

			
			-- Define the loss&gradient function, optimizer's state
			local function feval(optParams)
				gradParams:zero()

				-- Just run the two networks back and forth
				-- Params does not include the perception model's parameters
				-- as we don't need to train the perecption model
				local out_img = transfer_model:forward(content_img)
				for i=1,batch_size do
					out_img[i] = meansBRG(out_img[i]:double()):type(dtype)
				end
				perception_model:forward(out_img)
				local grad = perception_model:updateGradInput(out_img, dy)
				local loss = 0
			    for _, mod in ipairs(content_module) do
			      loss = loss + mod.loss
			      mod:setTarget(nil)
			    end
			    for _, mod in ipairs(style_module) do
			      loss = loss + mod.loss
			    end
				
				transfer_model:backward(content_img, grad)
				collectgarbage()
				return loss, gradParams
			end

			local optim_state = {learningRate = params.learning_rate}

			--for t = 1, params.iterations do
				x, losses = optim.adam(feval, optParams, optim_state)
				--print('Iteration number: '.. t ..'; Current loss: '.. losses[1])
			--end

		    num_iterations = num_iterations + 1--params.iterations	
			print('echo: ' .. nepoch .. '; Iteration number: '.. num_iterations ..'; Current loss: '.. losses[1])
			if(num_iterations % 200 == 0) then
				saveImage(transfer_model, num_iterations, testIm,params.style_img)
			end
			if(num_iterations % 10000 == 0) then
				print('Save params.trained_model')
				params.trained_model = build_modelname(params.style_img, nepoch, num_iterations)
				torch.save(params.trained_model, transfer_model)
			end
		end
	end
	params.trained_model = build_modelname(params.style_img, params.trained_epoch, 0)
	transfer_model:clearState()
	torch.save(params.trained_model, transfer_model)
end

local params = cmd:parse(arg)
main(params)
