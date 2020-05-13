def try_cuda(pytorch_obj):
  import torch.cuda
  try:
    disabled = torch.cuda.disabled
  except:
    disabled = False
  if torch.cuda.is_available() and not disabled:
    return pytorch_obj.cuda()
  else:
    return pytorch_obj
