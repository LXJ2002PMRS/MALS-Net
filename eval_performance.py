import time
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
from collections import OrderedDict
import network

class ThopWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, dem):
        feature = OrderedDict()
        feature['image'] = image
        feature['dem'] = dem
        return self.model(feature)


def evaluate_model_performance(model, image_tensor, dem_tensor, device='cuda'):

    model = model.to(device)
    
    image_tensor = image_tensor.to(device)
    dem_tensor = dem_tensor.to(device)

    print("FLOPs and Params...")
    wrapped_model = ThopWrapper(model).to(device)

    wrapped_model.eval() 
    macs, params = profile(wrapped_model, inputs=(image_tensor, dem_tensor), verbose=False)  
    flops = macs * 2 

    flops_str, params_str = clever_format([flops, params], "%.3f")

    feature = OrderedDict()
    feature['image'] = image_tensor
    feature['dem'] = dem_tensor
    model.eval()

    with torch.no_grad():
        for _ in range(50):
            _ = model(feature)

    iterations = 1000
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(feature)

    torch.cuda.synchronize()  
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_image = total_time / iterations
    fps = 1.0 / avg_time_per_image
    print("\n" + "=" * 55)
    print("Performance Evaluation Results")
    print("=" * 55)
    print(f"| Image Input Shape         | {str(tuple(image_tensor.shape)):<28} |")
    print(f"| DEM Input Shape           | {str(tuple(dem_tensor.shape)):<28} |")
    print(f"| Parameters                | {params_str:<28} |")
    print(f"| FLOPs                     | {flops_str:<28} |")
    print(f"| Avg Time                  | {avg_time_per_image * 1000:<7.2f} ms/iter          |")
    print(f"| FPS                       | {fps:<7.2f} frames/sec         |")
    print("=" * 55)

    return params, flops, fps

if __name__ == "__main__":
    model = network.deeplabv3plus_ECAResNet50(
        num_classes=1,    
        output_stride=16, 
    )
    B, C_img, H, W = 1, 1, 512, 512
    B, C_dem, H, W = 1, 1, 512, 512 

    dummy_image = torch.randn(B, C_img, H, W)
    dummy_dem = torch.randn(B, C_dem, H, W)

    evaluate_model_performance(model, dummy_image, dummy_dem, device='cuda')