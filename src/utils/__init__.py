def safe_get_source(sources):
    assert len(set(sources)) == 1
    return sources[0]


def adapt_voxels(voxels, training=False):
    if training:
        if voxels.dim() == 3 or voxels.dim() == 5:  # mindeye's dataloader
            repeat_index = random.randint(0, 2)
            voxels = voxels[:, repeat_index]
    else:
        if voxels.dim() == 3 or voxels.dim() == 5:
            voxels = voxels.mean(dim=1)
    return voxels
