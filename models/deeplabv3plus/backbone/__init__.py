from  models.deeplabv3plus.backbone import resnet, xception, drn, mobilenet
import timm

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'hrnet':
        return timm.create_model('hrnet_w64',features_only=True,pretrained=True, out_indices=(0,1,2,3,4))
    elif backbone == 'convnext':
        return timm.create_model('convnext_large_in22k',features_only=True,pretrained=True, num_classes=0, global_pool='',drop_path_rate=0.3)
        #convnext_large_384_in22ft1k
        #convnext_large_in22k
    else:
        raise NotImplementedError