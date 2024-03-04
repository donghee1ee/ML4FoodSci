import logging
import torch.distributed as dist

# TODO
def setup_logger():
    logger = logging.getLogger(__name__)
    dist.init_process_group(backend='nccl')
    if dist.get_rank() == 0:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    else:
        logging.disable(logging.CRITICAL)