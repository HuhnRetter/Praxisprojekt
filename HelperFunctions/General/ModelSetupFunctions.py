import torch
import torch.nn as nn
from torchvision import models

from HelperFunctions.DataFunctions.EnumsForParameter import PRETRAINED_MODEL, TRAINING_MODE


class JigsawNetwork(nn.Module):
    def __init__(self, num_classes=9, siamese_deg=9):
        super(JigsawNetwork, self).__init__()

        self.resnet = models.resnet18(pretrained=False)
        # remove fc layer
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.siamese_deg = siamese_deg

        if self.siamese_deg is None:
            self.fc = nn.Linear(512, num_classes)
        else:
            self.fc = nn.Linear(512 * self.siamese_deg, num_classes)

    def get_feature_vectors(self, input_batch):
        # Each input_batch would be of shape (batch_size, color_channels, h, w)
        x = self.resnet(input_batch)
        x = torch.flatten(x, 1)
        return x

    def forward(self, input_batch):

        # Data returned by data loaders is of the shape (batch_size, no_patches, h_patch, w_patch)
        # That's why named input to patches_batch

        if self.siamese_deg is None:
            final_feat_vectors = self.get_feature_vectors(input_batch)
            x = self.fc(final_feat_vectors)
        else:
            final_feat_vectors = None
            for patch_ind in range(self.siamese_deg):
                # Each patch_batch would be of shape (batch_size, color_channels, h_patch, w_patch)
                patch_batch = input_batch[:, patch_ind, :, :, :]
                patch_batch_features = self.get_feature_vectors(patch_batch)

                if patch_ind == 0:
                    final_feat_vectors = patch_batch_features
                else:
                    final_feat_vectors = torch.cat([final_feat_vectors, patch_batch_features], dim=1)
            x = self.fc(final_feat_vectors)
        return x


def rotNetSetup(num_classes_param):
    """adds a new Linear layer to a resnet18

    :param num_classes_param: number for trained classes
    :return: returns untrained rotnet (resnet18 + linear layer)
    """
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    # adding linear layer for rotation classification
    model.fc = nn.Linear(num_ftrs, num_classes_param)
    return model


def neuralNetSetup(PRETRAINEDFILE, num_classes_param, pretrained_model, param_freeze=True, param_eval=True):
    """loads one of the following pretrained models: rotnet, jigsaw, random and resnet18 for image classification
    :param PRETRAINEDFILE: path of the pretrained model to be loaded as a String
    :param num_classes_param: number for trained classes
    :param pretrained_model: enum class PRETRAINED_MODEL for choosing the specific model type
    :param param_freeze: for freezing the pretrained parameters
    :param param_eval: for changing the model into eval mode
    :return: returns a pretrained model resnet18
    """
    if pretrained_model == PRETRAINED_MODEL.ROTNET:
        model = models.resnet18(pretrained=False)

        # so not the whole neural net gets rebalanced
        if param_eval == True:
            model.eval()

        if param_freeze == True:
            print("Pretrained Params will be frozen")
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        # adding linear layer for rotation classification
        model.fc = nn.Linear(num_ftrs, 4)
        model.load_state_dict(torch.load(PRETRAINEDFILE))
        # print(f'{model}\n\n')

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes_param)
        # print(f'{model}\n\n')
        return model

    elif pretrained_model == PRETRAINED_MODEL.JIGSAW:
        model = JigsawNetwork(num_classes_param, None)
        model.fc = nn.Linear(512 * 9, 100)
        # print(model)
        print("#####################\n\n")
        model = load_model(model, PRETRAINEDFILE)

        if param_eval == True:
            model.eval()

        if param_freeze == True:
            print("Pretrained Params will be frozen")
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(512, 3)
        # print(model)
        return model

    elif pretrained_model == PRETRAINED_MODEL.RESNET18:
        model = models.resnet18(pretrained=True)

        if param_eval == True:
            model.eval()
        # so not the whole neural net gets rebalanced
        if param_freeze == True:
            print("Pretrained Params will be frozen")
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes_param)
        return model

    elif pretrained_model == PRETRAINED_MODEL.RANDOM:
        model = models.resnet18(pretrained=False)

        if param_eval == True:
            model.eval()
        # so not the whole neural net gets rebalanced
        if param_freeze == True:
            print("Pretrained Params will be frozen")
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes_param)
        return model


def load_model(model, FILE, training_mode=TRAINING_MODE.NORMAL):
    """loads an initialized model from a given path

    :param model: initialized model
    :param FILE: path of the model to be loaded as a String
    :return: returns the fully loaded model
    """
    model.to(torch.device("cpu"))
    model.load_state_dict(torch.load(FILE))
    if training_mode == TRAINING_MODE.LOAD_EXISTING_MODEL_TESTING:
        model.eval()

    if training_mode == TRAINING_MODE.LOAD_EXISTING_MODEL_TRAINING:
        print("Pretrained Params will be unfrozen")
        for param in model.parameters():
            param.requires_grad = True
    return model
