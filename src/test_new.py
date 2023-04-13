import torch
list_t_1 = [torch.full(size=(1,3,1,5,5), fill_value=i) for i in range(5)]
tensor_1 = torch.cat([torch.full(size=(1,3,1,5,5), fill_value=i) for i in range(5,10)], dim=1)
tensor_2 = torch.cat(list_t_1, dim=1)
tensor = torch.cat([tensor_2, tensor_1], dim=0)
print(tensor)
n = tensor_2.chunk(5,1)
m = torch.cat(n,2)

num_predicted_frames = 5
x = torch.randn(size=(8,256,16,16))

conv = torch.nn.Conv2d(256, 256 * num_predicted_frames, 3,1,1)
x_1 = torch.cat(conv(x).unsqueeze(2).chunk(5, 1), 2)
print(x_1)