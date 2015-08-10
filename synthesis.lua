require 'torch'
require 'image'


local function extract_neighborhood(X, k, i, j, normalize, neigh)
  --[[
  Extract a neighborhood of pixels from an image, flattening it and possibly
  normalizing it.
  
  Inputs:
  - X: Tensor of shape C x H x W
  - k: int giving kernel size; must be odd
  - i, j: row and column of X giving the center of the neighborhood
  - normalize: Whether to L2 normalize the neighborhood; default is true
  - neigh: Tensor to store the results; if not given, a new Tensor is returned.
  --]]
  assert(k % 2 == 1)
  normalize = normalize or true
  local X_chunk = X[{{}, {i - k + 1, i - 1}, {j - (k - 1) / 2, j + (k - 1) / 2}}]
  local X_row = X[{{}, {i, i}, {j - (k - 1) / 2, j - 1}}]
  local s1, s2 = X_chunk:nElement(), X_row:nElement()
  
  local X_chunk_flat = X_chunk:contiguous():view(X_chunk:nElement())
  local X_row_flat = X_row:contiguous():view(X_row:nElement())

  if neigh then
    assert(neigh:nDimension() == 1)
    assert(neigh:nElement() == s1 + s2)
  else
    neigh = X.new(s1 + s2)
  end
  neigh[{{1, s1}}] = X_chunk_flat
  neigh[{{s1 + 1, s1 + s2}}] = X_row_flat
  if normalize then
    neigh:div(torch.norm(neigh))
  end
  return neigh
end


local function extract_all_neighborhoods(X, k, normalize)
  --[[
  Extract all neighborhoods from an image as a big matrix. Also return a
  flattened version of the input making it easy to map from neighborhoods back
  to pixel values at their center.
  
  Inputs:
  - X: Tensor of shape C x H x W
  - k: Kernel size; must be an odd integer
  - normalize: Whether to normalize the neighborhoods; default is true.
  
  Returns:
  - mat: Tensor of shape N x S where N is the total number of neighborhoods in
    the image and S is the size of each neighborhood.
  - vals: Tensor of shape N x C giving flattened version of X; in particular,
    if a neighborhood is centered at (i, j) in X and ends up as the kth row of
    mat then vals[k] == X[{{}, i, j}].
  --]]
  assert(k % 2 == 1)
  local Hs, Ws = X:size(2), X:size(3)
  local mat = nil
  local vals = nil
  local idx = 1
  for i = k, Hs do
    local j_start = 1 + (k - 1) / 2
    local j_end = Ws - (k - 1) / 2
    for j = j_start, j_end do
      local neigh = extract_neighborhood(X, k, i, j, normalize)
      if not mat then
        local num_rows = (Hs - k + 1)
        local num_cols = (j_end - j_start + 1)
        local num_neigh = num_rows * num_cols
        mat = neigh.new(num_neigh, neigh:nElement())
        vals = X.new(num_neigh, X:size(1))
      end
      mat[{idx, {}}] = neigh
      vals[{idx, {}}] = X[{{}, i, j}]
      idx = idx + 1
    end
  end
  return mat, vals
end


local function initialize_image(X_src, H, W, k)
  --[[
  Initialize an image for texture synthesis by initializing its top, left, and
  right borders with random pixels from a source image.

  Inputs:
  - X_src: Tensor of shape C x HH x WW giving source image
  - H, W: Desired height and width of generated image.
  - k: kernel size that will be used for texture synthesis.

  Returns:
  - X: Tensor of shape C x (H + k - 1) x (W + k - 1) where the top k -1 rows and
    leftmost and rightmost (k - 1) / 2 columns have been initialized with random
    elements from X_src.
  --]]
  local X = X_src.new(X_src:size(1), H + k - 1, W + k - 1):zero()
  
  local function random_src_color()
    local y = torch.random(1, X_src:size(2))
    local x = torch.random(1, X_src:size(3))
    return X_src[{{}, y, x}]
  end
  
  for i = 1, k - 1 do
    for j = 1, X:size(3) do
      X[{{}, i, j}] = random_src_color()
    end
  end
  
  for i = k, X:size(2) do
    for j = 1, (k - 1) / 2 do
      X[{{}, i, j}] = random_src_color()
      X[{{}, i, X:size(3) - j + 1}] = random_src_color()
    end
  end
  
  return X
end


local function synthesize_texture(X_src, H, W, k, gpu)
  --[[
  Use the source image X_src to synthesize a texture of size H x W, using a
  kernel size of k.

  Inputs:
  - X_src: Source image; Tensor of shape C x HH x WW
  - H, W: Height and width of the texture to generate
  - k: Kernel size to use for synthesis; must be an odd integer
  - gpu: Boolean, whether or not to use GPU

  Returns:
  - X: Tensor of shape C x H x W giving generated texture.
  --]]
  local X = initialize_image(X_src, H, W, k)
  local mat, vals = extract_all_neighborhoods(X_src, k)
  if gpu then
    mat = mat:cuda()
  end

  -- sims will store the similarity of the current neighborhood and all
  -- neighborhoods; reusing it avoids GPU allocations in the hot loop.
  -- Similarly neigh will store the current neighborhood of X.
  local sims = mat.new(mat:size(1))
  local neigh = X.new(mat:size(2))
  local neigh_gpu = nil
  if gpu then
    neigh_gpu = neigh:cuda()
  end

  -- Precompute a couple constants
  local num_rows = X:size(2) - k + 1
  local j_start = 1 + (k - 1) / 2
  local j_end = X:size(3) - (k - 1) / 2

  -- Synthesize the texture, one pixel at a time
  for i = k, X:size(2) do
    local row_num = i - k + 1
    if row_num % 20 == 0 then
      print(string.format('Starting row %d / %d', row_num, num_rows))
    end
    for j = j_start, j_end do
      extract_neighborhood(X, k, i, j, false, neigh)
      if gpu then
        neigh_gpu:copy(neigh)
        sims:mv(mat, neigh_gpu)
      else
        sims:mv(mat, neigh)
      end
      local max_sim, max_idx = torch.max(sims:float(), 1)
      X[{{}, i, j}] = vals[max_idx[1]]
    end
  end

  -- Chop out the bits that contain the random initialization
  local y0, y1 = k, X:size(2)
  local x0, x1 = 1 + (k - 1) / 2, X:size(3) - (k - 1) / 2
  return X[{{}, {y0, y1}, {x0, x1}}]
end


local cmd = torch:CmdLine()
cmd:option('-source', 'examples/inputs/scales.png',
    'Source image for texture synthesis')
cmd:option('-output_file', 'out.png', 'Filename of synthesized texture')
cmd:option('-height', 64, 'Height of generated image')
cmd:option('-width', 64, 'Width of generated image')
cmd:option('-k', 15, 'Kernel size to use for texture synthesis')
cmd:option('-gpu', 0,
    'Zero-indexed ID of the GPU to use; to use CPU only, set -gpu < 0')
local params = cmd:parse(arg)

-- Handle GPU stuff
if params.gpu >= 0 then
  require 'cutorch'
  cutorch.setDevice(params.gpu + 1)
end

local X_src = image.load(params.source)
local gpu = params.gpu >= 0
local H, W = params.height, params.width
local X = synthesize_texture(X_src, H, W, params.k, gpu)
image.save(params.output_file, X)

