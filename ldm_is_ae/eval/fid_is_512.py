#!/usr/bin/env python3
# fid_is_512.py -- 512-resolution entry point for the FID/IS evaluator in
# ldm_is_ae/eval/fid_is.py.  Same implementation, same caliber (torch-fidelity Inception features,
# ADM/TF Inception Score); the 512 line only differs in the reference statistics, which are
# the 512 ones (jit_in512_stats.npz) instead of the 256 ones.
#
# Usage:
#   python3 fid_is_512.py --sample_path <gen_dir> [--ref_stats <jit_in512_stats.npz>] [options]
#   --ref_stats defaults to $JIT_IN512_STATS, else to <repo>/fid_stats/jit_in512_stats.npz
#   when that file exists.  All other options are those of ldm_is_ae/eval/fid_is.py
#   (--tag --out_json --weights --batch_size --device).
import os, sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)

import fid_is  # noqa: E402  (same directory)

DEFAULT_REF = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'fid_stats', 'jit_in512_stats.npz')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a == '--ref_stats' or a.startswith('--ref_stats=') for a in argv):
        ref = os.environ.get('JIT_IN512_STATS') or DEFAULT_REF
        if os.path.exists(ref):
            argv += ['--ref_stats', ref]
            print('fid_is_512: reference statistics = %s' % ref, flush=True)
        else:
            print('fid_is_512: no reference statistics found -- pass --ref_stats (e.g. the ADM '
                  'VIRTUAL_imagenet512.npz) or place jit_in512_stats.npz at %s' % DEFAULT_REF,
                  file=sys.stderr)
    return fid_is.main_cli(argv)


if __name__ == '__main__':
    main()
