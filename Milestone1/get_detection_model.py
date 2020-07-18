import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def get_detection_model(num_classes, model='mobilenet', anch='15-highersize'):

  """
  - model --> "mobilenet" for a MobileNet backbone architecture
          --> "VGG" for a VGG 16 backbone architecture
  - anch --> "15-highersize" anchors with sizes ((32, 64, 128, 256, 512),)
         --> "15-lowersize" anchors with sizes ((8, 16, 32, 64, 128),)
         --> "9" anchors with sizes ((8, 16, 32,),)
  """

  # load a pre-trained model for classification and return only the features
  if model == 'mobilenet' :
    # mobilenet backbone
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280 # vgg16: 512, mobilenet_v2: 1280, resnet50: 256
  elif model == 'VGG' :
    # vgg16 backbone
    backbone = torchvision.models.vgg16(pretrained=True).features
    backbone.out_channels = 512   # vgg16: 512, mobilenet_v2: 1280, resnet50: 256

  # Anchor generation
  # RPN generate X x Y anchors per spatial location, with X different sizes and 
  # Y different aspect ratios (initially)
  if anch == '15-highersize' :
    # X=5, Y=3 --> 15 anchors
    anchor_sizes = ((32, 64, 128, 256, 512),)
    ratios = ((0.5, 1.0, 2.0),)
  elif anch == '15-lowersize' :
    # X=5, Y=3 --> 15 anchors (lower size) 
    anchor_sizes = ((8, 16, 32, 64, 128),)
    ratios = ((0.5, 1.0, 2.0),)
  elif anch == '9' :
    # X=3, Y=3 --> 9 anchors
    anchor_sizes = ((8, 16, 32,),)
    ratios = ((0.5, 1.0, 2.0),)

  anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                    aspect_ratios=ratios)
  
  # feature maps for the region of interest cropping, as well as
  # the size of the crop after rescaling.
  roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                  output_size=7,
                                                  sampling_ratio=2)
  
  model = FasterRCNN(backbone,
                    num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
  return model