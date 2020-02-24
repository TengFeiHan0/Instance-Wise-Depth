

# local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern
from defaults import get_cfg_defaults 
 
if __name__ == "__main__":
  cfg = get_cfg_defaults()
  cfg.merge_from_file("instance_depth.yaml")
  cfg.freeze()
  print(cfg)
