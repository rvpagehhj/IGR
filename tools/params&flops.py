import _init_paths
import _submodules.neuron.neuron.data as data
from trackers import *
from trackers.ostrack import OneShotTrack
import torch
from thop import profile
from thop.utils import clever_format
import time

if __name__ == '__main__':
    cfg_file = '/home/sys123/yh/workspace/configs/config.py'
    ckp_file = '/home/sys123/yh/workspace/tools/work_dirs/ATSS res 24/latest.pth'
    transforms = data.BasicPairTransforms(train=False)
    tracker = OneShotTrack(
        cfg_file, ckp_file, transforms,
        name_suffix='ATSS')

    device = "cuda:0"
    torch.cuda.set_device(device)
    '''Speed Test'''
    search = torch.randn(1,1,3,1333,800).to(device)
    tracker.model.forward = tracker.model.forward_dummy
    macs1, params1 = profile(tracker.model, inputs=search,
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)
    T_w = 50
    T_t = 100
    print("testing speed ...")
    search = torch.randn(1, 3, 1333, 800).to(device)
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = tracker.model(search)
        start = time.time()
        for i in range(T_t):
            _ = tracker.model(search)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))

