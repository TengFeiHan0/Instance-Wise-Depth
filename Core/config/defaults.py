import os 

from yacs.config import CfgNode as CN 

_C = CN()
_C.WEIGHT = ""
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_EPOCHS=50
_C.MODEL.MASK_LR=0.01
_C.MODEL.MASK_WEIGHT_DECAY= 0.0005
_C.MODEL.NAME="instanceDepth"

# ---------------------------------------------------------------------------- #
# EmbedMask Options
# ---------------------------------------------------------------------------- #
_C.MODEL.EMBED_MASK = CN()
_C.MODEL.EMBED_MASK.NUM_CLASSES = 9  # the number of classes including background
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
_C.MODEL.EMBED_MASK.MASK_NUM_CONVS = 4
# when input box to predict margin, the box should be scaled. if the scale equals to -1, means use the shorter side length
_C.MODEL.EMBED_MASK.BOX_TO_MARGIN_SCALE = 800
# blocking the gradient from box to margin may bring better performance
_C.MODEL.EMBED_MASK.BOX_TO_MARGIN_BLOCK = True
_C.MODEL.EMBED_MASK.FPN_POST_NMS_TOP_N = 100
_C.MODEL.EMBED_MASK.MIN_SIZE = 0

# inference settings
_C.MODEL.EMBED_MASK.MASK_TH = 0.5
_C.MODEL.EMBED_MASK.POSTPROCESS_MASKS = False
_C.MODEL.EMBED_MASK.IN_CHANNEL = 256

#-----FPN---------------------
_C.MODEL.FPN = CN()
_C.MODEL.FPN.IN_CHANNELS = [256, 512, 1024, 2048]
_C.MODEL.FPN.OUTPUT_CHANNEL = 256
_C.MODEL.FPN.USE_P5 = True

#  depth --------------------------------------------------------------
_C.MODEL.DEPTH = CN()
_C.MODEL.DEPTH.SCALES= [0, 1, 2, 3]
_C.MODEL.DEPTH.FRAME_IDS= [0, -1, 1]
_C.MODEL.DEPTH.POSE_FRAMES_INPUT= "pair"
_C.MODEL.DEPTH.NUM_LAYERS= 50
_C.MODEL.DEPTH.POSE_MODEL_TYPE = "shared"
_C.MODEL.DEPTH.LEARNING_RATE = 1e-4
_C.MODEL.DEPTH.SCHEDULER_STEP_SIZE = 15
_C.MODEL.DEPTH.NUM_CH_ENC = [64, 64, 128, 256, 512]
_C.MODEL.DEPTH.NUM_CH_DEC = [64, 64, 128, 256, 512]
_C.MODEL.DEPTH.USE_SKIPS= True
_C.MODEL.DEPTH.NUM_OUTPUT_CHANNELS= 1
_C.MODEL.DEPTH.AVG_REPROJECTION=False
_C.MODEL.DEPTH.SMOOTH_WEIGHT=0.001
_C.MODEL.DEPTH.NUM_SCALES = 4
_C.MODEL.DEPTH.MIN_DEPTH = 0.1
_C.MODEL.DEPTH.MAX_DEPTH = 100.0
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800, )  # (800,)
# The range of the smallest side for multi-scale training
_C.INPUT.MIN_SIZE_RANGE_TRAIN = (-1, -1)  # -1 means disabled and it will use MIN_SIZE_TRAIN
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True
_C.INPUT.HEIGHT = 1024
_C.INPUT.WIDTH = 2048

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ("cityscapes_mask_instance_train",)
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ("cityscapes_mask_instance_val",)
_C.DATASETS.TRAIN_PATH = "./datasets/"
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 32
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2
# the learning rate factor of deformable convolution offsets
_C.SOLVER.DCONV_OFFSETS_LR_FACTOR = 1.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 2500

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 8

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 8
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")



def get_cfg_defaults():
    return _C.clone()