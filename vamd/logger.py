import yaml, logging, socket, os, sys

model_dir = os.environ.get("MODEL_DIR", "./workdir/default")

def get_logger(name):
    logger = logging.Logger(name)
    level = {"crititical": 50, "error": 40, "warning": 30, "info": 20, "debug": 10}[
        os.environ.get("LOGGER_LEVEL", "info")
    ]
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    os.makedirs(model_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(model_dir, "log.out"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        f"%(asctime)s [{socket.gethostname()}:%(process)d] [%(levelname)s] %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def init():
    if bool(int(os.environ.get("WANDB_LOGGING", "0"))):
        os.makedirs(model_dir, exist_ok=True)
        out_file = open(os.path.join(model_dir, "std.out"), 'ab')
        os.dup2(out_file.fileno(), 1)
        os.dup2(out_file.fileno(), 2)
    
init()