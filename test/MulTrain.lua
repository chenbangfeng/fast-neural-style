require 'lfs'
require 'os'


local cmd = torch.CmdLine()
cmd:option('-style_path', 'style_images', 'Style image to train on the image set')



local function main(params)
	style_img_files = {}
	file_num = 1
	for file in lfs.dir(params.style_path) do
        if file ~= "." and file ~= ".." then
            local f = params.style_path..'/'..file
            local attr = lfs.attributes (f)
            assert (type(attr) == "table")
            if attr.mode ~= "directory" then
				style_img_files[file_num] = f
				file_num = file_num + 1
			end
		end
	end

	for i=1,#style_img_files do
		print("train style<"  .. style_img_files[i] .. ">")
		os.execute("th train.lua " .. "-style_img " .. style_img_files[i])
	end

end


local params = cmd:parse(arg)
main(params)