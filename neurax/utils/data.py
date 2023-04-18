def batch_uniform_sequence(seq, size, skip_start=0, skip_end=0, pre_ctx=0, post_ctx=0):
    n_samples = len(seq) - skip_start - skip_end
    n_trials = n_samples // size
    assert n_samples % size == 0
    assert 0 <= pre_ctx <= skip_start
    assert 0 <= post_ctx <= skip_end

    return [
        seq[skip_start+i*size-pre_ctx:skip_start+(i+1)*size+post_ctx]
        for i in range(n_trials)
    ]

