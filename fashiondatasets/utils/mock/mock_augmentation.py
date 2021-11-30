from fashiondatasets.utils.logger import defaultLogger


def pass_trough():
    defaultLogger().warning("WARNING! Using Pass Through Augmentation for Dev!")
    return lambda d: d
