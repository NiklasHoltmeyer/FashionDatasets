from fashiondatasets.utils.logger.defaultLogger import defaultLogger


def pass_trough():
    defaultLogger().warning("WARNING! Using Pass Through Augmentation for Dev!")
    return lambda d: d