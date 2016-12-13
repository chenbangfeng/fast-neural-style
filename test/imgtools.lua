require 'image'


function pp(im)
	-- Take in RGB and subtract means
	-- If you're wondering what these floats are for, like I did, they're the mean pixel (Blue-Green-Red) for the pretrained VGG net in order to zero-center the input image.
	local means = torch.DoubleTensor({-123.68, -116.779, -103.939})
	--Convert RGB --> BGR and pixel range to 0-255
	im = im:mul(255.0)
	means = means:view(3, 1, 1):expandAs(im)
	-- Subtract means and resize
	im:add(1, means)
	return im
end

function means(im)
	local means = torch.DoubleTensor({-123.68, -116.779, -103.939})
	means = means:view(3, 1, 1):expandAs(im)
	im:add(1, means)
	return im
end

function toBGR(im)
	local perm = torch.LongTensor{3, 2, 1}
	--Convert RGB --> BGR and pixel range to 0-255
	im = im:index(1, perm)
	return im

end

function toBRG(im)
	local perm = torch.LongTensor{3, 1, 2}
	--Convert RGB --> BGR and pixel range to 0-255
	im = im:index(1, perm)
	return im

end

function scale_pp(im,size)
	-- Take in grayscale/RGB return RGB-type scaled image
	if size then
		if im:size(1) ==1 then
			im = torch.cat(im,torch.cat(im,im,1),1)
		elseif im:size(1) == 3 then
		else
			error('Input image is not an RGB image')
		end
		im = image.scale(im,size,size,'bicubic')
	end
	return im
	
end

function scale_crop(img, size)
    h = img:size(2)
    w = img:size(3)
    if w < h then
        if w < size then
            img = image.scale(img, size, size*h/w,'bilinear')
            h = img:size(2)
            w = img:size(3)
        end
    else
        if h < size then
            img = image.scale(img, size*w/h, size,'bilinear')
            h = img:size(2)
            w = img:size(3)
        end
    end
    img = image.crop(img,(w-size)*0.5, (h-size)*0.5, (w+size)*0.5, (h+size)*0.5)
    return img
end


function ppBRG(im)
	-- Take in RGB and subtract means
	-- If you're wondering what these floats are for, like I did, they're the mean pixel (Blue-Red-Green) for the pretrained VGG net in order to zero-center the input image.
	local means = torch.DoubleTensor({ -103.939, -123.68, -116.779})
	--Convert RGB --> BGR and pixel range to 0-255
	im = im:mul(255.0)
	means = means:view(3, 1, 1):expandAs(im)
	-- Subtract means and resize
	im:add(1, means)
	return im
end

function ppBGR(im)
	-- Take in RGB and subtract means
	-- If you're wondering what these floats are for, like I did, they're the mean pixel (Blue-Green-Red) for the pretrained VGG net in order to zero-center the input image.
	local means = torch.DoubleTensor({ -103.939, -116.779, -123.68})
	--Convert RGB --> BGR and pixel range to 0-255
	im = im:mul(255.0)
	means = means:view(3, 1, 1):expandAs(im)
	-- Subtract means and resize
	im:add(1, means)
	return im
end



function BRGtoRGB(im)
	local perm = torch.LongTensor{2, 3, 1}
	--Convert BRG --> RGB and pixel range to 0-255
	im = im:index(1, perm)
	return im

end

function BGRtoRGB(im)
	local perm = torch.LongTensor{3, 2, 1}
	--Convert BGR --> RGB and pixel range to 0-255
	im = im:index(1, perm)
	return im

end

function meansBRG(im)
	local means = torch.DoubleTensor({ -103.939, -123.68, -116.779})
	means = means:view(3, 1, 1):expandAs(im)
	im:add(1, means)
	return im
end

function meansBGR(im)
	local means = torch.DoubleTensor({ -103.939, -116.779, -123.68})
	means = means:view(3, 1, 1):expandAs(im)
	im:add(1, means)
	return im
end

function dp(im)
	-- Exact inverse of above
	local perm = torch.LongTensor{3, 2, 1}
	im = im:index(1, perm)
	return im:double()
end

local vgg_mean = torch.DoubleTensor({103.939, 116.779, 123.68})

--[[
Preprocess an image before passing to a VGG model. We need to rescale from
[0, 1] to [0, 255], convert from RGB to BGR, and subtract the mean.

Input:
- img: Tensor of shape (N, C, H, W) giving a batch of images. Images 
]]
function preprocess(img)
  local mean = vgg_mean:view(3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return img:index(1, perm):mul(255):add(-1, mean)
end


-- Undo VGG preprocessing
function deprocess(img)
  local mean = vgg_mean:view(3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return (img + mean):div(255):index(1, perm)
end
