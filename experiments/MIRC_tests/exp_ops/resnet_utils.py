import models

def interpret_resnet(resnet_type):
    if resnet_type == 50:
        spec = models.get_data_spec(model_class=models.attResNet50)
        net = models.ResNet50
    elif resnet_type == 101:
        spec = models.get_data_spec(model_class=models.attResNet101)
        net = models.ResNet101
    elif resnet_type == 152:
        spec = models.get_data_spec(model_class=models.attResNet152)
        net = models.ResNet152
    return net, spec