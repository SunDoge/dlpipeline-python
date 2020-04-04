import pyvips
import logging

logger = logging.getLogger(__name__)

def pyvips_loader(path: str, access=pyvips.Access.SEQUENTIAL):
    return pyvips.Image.new_from_file(path, access=access)


