from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
# _C.MODEL.FCOS = CN()

# # This is the number of foreground classes.
# _C.MODEL.FCOS.NUM_CLASSES = 80
# _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
# _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
# _C.MODEL.FCOS.PRIOR_PROB = 0.01
# _C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
# _C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
# _C.MODEL.FCOS.NMS_TH = 0.6
# _C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
# _C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
# _C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
# _C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
# _C.MODEL.FCOS.TOP_LEVELS = 2
# _C.MODEL.FCOS.NORM = "GN"  # Support GN or none
# _C.MODEL.FCOS.USE_SCALE = True

# # Multiply centerness before threshold
# # This will affect the final performance by about 0.05 AP but save some time
# _C.MODEL.FCOS.THRESH_WITH_CTR = False

# # Focal loss parameters
# _C.MODEL.FCOS.LOSS_ALPHA = 0.25
# _C.MODEL.FCOS.LOSS_GAMMA = 2.0
# _C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
# _C.MODEL.FCOS.USE_RELU = True
# _C.MODEL.FCOS.USE_DEFORMABLE = False

# # the number of convolutions used in the cls and bbox tower
# _C.MODEL.FCOS.NUM_CLS_CONVS = 4
# _C.MODEL.FCOS.NUM_BOX_CONVS = 4
# _C.MODEL.FCOS.NUM_SHARE_CONVS = 0
# _C.MODEL.FCOS.CENTER_SAMPLE = True
# _C.MODEL.FCOS.POS_RADIUS = 1.5
# _C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'

# ---------------------------------------------------------------------------- #
# embedmask Head
# ---------------------------------------------------------------------------- #
_C.MODEL.EMBED_MASK = CN()

_C.MODEL.EMBED_MASK.NUM_CLASSES = 80  # the number of classes including background
#_C.MODEL.EMBED_MASK.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.EMBED_MASK.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT = (64, 128, 256, 512)
_C.MODEL.EMBED_MASK.PRIOR_PROB = 0.01
_C.MODEL.EMBED_MASK.INFERENCE_TH = 0.05
_C.MODEL.EMBED_MASK.NMS_TH = 0.6
_C.MODEL.EMBED_MASK.PRE_NMS_TOP_N = 1000
# Focal loss parameter: alpha
_C.MODEL.EMBED_MASK.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.EMBED_MASK.LOSS_GAMMA = 2.0
_C.MODEL.EMBED_MASK.HEAD_NUM_CONVS = 4
_C.MODEL.EMBED_MASK.NORM_REG_TARGETS = True
_C.MODEL.EMBED_MASK.CENTERNESS_ON_REG = True
_C.MODEL.EMBED_MASK.USE_DCN_IN_TOWER = False
_C.MODEL.EMBED_MASK.IOU_LOSS_TYPE = "giou"
_C.MODEL.EMBED_MASK.CENTER_ON = True
_C.MODEL.EMBED_MASK.CENTER_POS_RADIOS = 1.5
#------Embedding Options-------#
_C.MODEL.EMBED_MASK.EMBED_DIM = 32

# box mask select related settings
_C.MODEL.EMBED_MASK.SAMPLE_IN_MASK = True
_C.MODEL.EMBED_MASK.SAMPLE_POS_IOU_TH = 0.5
_C.MODEL.EMBED_MASK.FIX_MARGIN = False

_C.MODEL.EMBED_MASK.BOX_PADDING = 0.0

_C.MODEL.EMBED_MASK.PRIOR_MARGIN = 2.0

_C.MODEL.EMBED_MASK.LOSS_MASK_ALPHA = 0.5
_C.MODEL.EMBED_MASK.LOSS_SMOOTH_ALPHA = 0.1

_C.MODEL.EMBED_MASK.MASK_SCALE_FACTOR = 2
_C.MODEL.EMBED_MASK.NUM_MASK_CONVS =4
_C.MODEL.EMBED_MASK.NUM_CLS_CONVS  =4
_C.MODEL.EMBED_MASK.NUM_BOX_CONVS  =4
_C.MODEL.EMBED_MASK.NUM_SHARE_CONVS=0

# when input box to predict margin, the box should be scaled. if the scale equals to -1, means use the shorter side length
_C.MODEL.EMBED_MASK.BOX_TO_MARGIN_SCALE = 800
# blocking the gradient from box to margin may bring better performance
_C.MODEL.EMBED_MASK.BOX_TO_MARGIN_BLOCK = True

# inference settings
_C.MODEL.EMBED_MASK.MASK_TH = 0.5
_C.MODEL.EMBED_MASK.POSTPROCESS_MASKS = False
