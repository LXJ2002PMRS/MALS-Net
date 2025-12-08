from .deeplabv3plus import DeepLabV3, DeepLabHeadV3Plus_cat_eca_Resnet50
from .backbone import Cat_eca_resnetv6
from torchvision.models._api import register_model

@register_model()
def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]


    backbone = Cat_eca_resnetv6.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    inplanes = 2048
    low_level_planes = 256

    if name == 'deeplabv3plus':
        classifier = DeepLabHeadV3Plus_cat_eca_Resnet50(inplanes, low_level_planes, num_classes, aspp_dilate)
    model = DeepLabV3(backbone, classifier)
    return model

@register_model()
def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if backbone.startswith('cat_eca_resnet50'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                             pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


# Deeplab v3+
@register_model()
def deeplabv3plus_ECAResNet50(num_classes=1, output_stride=16, pretrained_backbone=False):
    """Constructs a DeepLabV3 model with a ECA-ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'cat_eca_resnet50', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)