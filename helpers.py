import os

def get_prefix(args):
    checkpoint_dir = args.save_path + 'checkpoint-' + args.description
    return os.path.join(checkpoint_dir, 'ckpt'), checkpoint_dir