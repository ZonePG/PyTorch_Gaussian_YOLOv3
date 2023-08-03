import torch
import torch.nn as nn

from collections import defaultdict
from models.mmyolo_layer import MMYOLOLayer
from models.yolov3 import create_yolov3_modules


class MMYOLOv3(nn.Module):
    """
    multimodal yolov3
    """

    def __init__(self, config_model, ignore_thre=0.7):
        """
        Initialization of YOLOv3 class.
        Args:
            config_model (dict): used in YOLOLayer.
            ignore_thre (float): used in YOLOLayer.
        """
        super(MMYOLOv3, self).__init__()

        if config_model["TYPE"] == "YOLOv3":
            self.module_list_0 = create_yolov3_modules(config_model, ignore_thre)
            self.module_list_0.pop(-1)
            print(self.module_list_0)
            self.module_list_1 = create_yolov3_modules(config_model, ignore_thre)
            self.module_list_1.pop(-1)
            self.mmyolo_layer = MMYOLOLayer(config_model, layer_no=2, in_ch=256, ignore_thre=ignore_thre)
        else:
            raise Exception(
                "Model name {} is not available".format(config_model["TYPE"])
            )

    def forward(self, x0, x1, targets=None):
        """
        Forward path of YOLOv3.
        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`

        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """
        train = targets is not None
        output = []
        self.loss_dict = defaultdict(float)
        route_layers = []
        for i, module in enumerate(self.module_list_0):
            # yolo layers
            if i in [14, 22, 28]:
                if train:
                    x0, *loss_dict = module(x0, targets)
                    for name, loss in zip(["xy", "wh", "conf", "cls"], loss_dict):
                        self.loss_dict[name] += loss
                else:
                    x0 = module(x0)
                output.append(x0)
            else:
                x0 = module(x0)

            # route layers
            if i in [6, 8, 12, 20]:
                route_layers.append(x0)
            if i == 14:
                x0 = route_layers[2]
            if i == 22:  # yolo 2nd
                x0 = route_layers[3]
            if i == 16:
                x0 = torch.cat((x0, route_layers[1]), 1)
            if i == 24:
                x0 = torch.cat((x0, route_layers[0]), 1)

        route_layers.clear()
        for i, module in enumerate(self.module_list_1):
            # yolo layers
            if i in [14, 22, 28]:
                if train:
                    x1, *loss_dict = module(x1, targets)
                    for name, loss in zip(["xy", "wh", "conf", "cls"], loss_dict):
                        self.loss_dict[name] += loss
                else:
                    x1 = module(x1)
                output.append(x1)
            else:
                x1 = module(x1)

            # route layers
            if i in [6, 8, 12, 20]:
                route_layers.append(x1)
            if i == 14:
                x1 = route_layers[2]
            if i == 22:  # yolo 2nd
                x1 = route_layers[3]
            if i == 16:
                x1 = torch.cat((x1, route_layers[1]), 1)
            if i == 24:
                x1 = torch.cat((x1, route_layers[0]), 1)

        # i == 28
        if train:
            x, *loss_dict = self.mmyolo_layer(x0, x1, targets)
            for name, loss in zip(["xy", "wh", "conf", "cls"], loss_dict):
                self.loss_dict[name] += loss
        else:
            x = self.mmyolo_layer(x0, x1)
        output.append(x)

        if train:
            return sum(output)
        else:
            return torch.cat(output, 1)
