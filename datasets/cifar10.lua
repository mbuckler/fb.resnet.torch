--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  CIFAR-10 dataset loader
--

local t = require 'datasets/transforms'

local M = {}
local CifarDataset = torch.class('resnet.CifarDataset', M)

function CifarDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
end

function CifarDataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function CifarDataset:size()
   return self.imageInfo.data:size(1)
end

-- Computed from entire CIFAR-10 training set
--local meanstd = {
--   mean = {125.3, 123.0, 113.9},
--   std  = {63.0,  62.1,  66.7},
--}

function CifarDataset:preprocess()

   local r_mean, g_mean, b_mean, r_std,  g_std,  b_std
   local file = io.open("mean_std.txt")

   if file then
     for line in file:lines() do
        r_mean,g_mean,b_mean,r_std,g_std,b_std = unpack(line:split(" "))
     end
   else
     error('Cannot find mean and std file');
   end

   local meanstd = {
     mean = {tonumber(r_mean), tonumber(g_mean), tonumber(b_mean)},
     std  = {tonumber(r_std ), tonumber(g_std ), tonumber(b_std )},
   }

   if self.split == 'train' then
      return t.Compose{
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
         t.RandomCrop(32, 4),
      }
   elseif self.split == 'val' then
      return t.ColorNormalize(meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.CifarDataset
