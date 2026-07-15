"""
Precoder selection. The default wideband mode accumulates the channel
covariance over all active subcarriers, takes the eigendecomposition of
the symmetrized covariance, and returns the dominant eigenvectors as
unit-norm precoder columns; one matrix serves every subcarrier, so the
effective channel remains as smooth in frequency as the physical channel,
which is the condition the interpolating least-squares DM-RS estimator
requires. On a flat channel the wideband solution equals the per-subcarrier singular value
decomposition. The per-subcarrier mode maximizes
beamforming gain but carries an arbitrary phase per subcarrier and must
only be used with ideal channel state information on selective channels.
The identity mode is unprecoded spatial multiplexing, kept for the
equalizer comparison: under any eigenbeamforming precoder the
effective Gram matrix is diagonal on a flat channel and ZF, MMSE, and
unbiased MMSE provably coincide, so the equalizer comparison requires the
unprecoded general channel. The maximum-ratio mode returns the dominant
wideband eigenvector for single-layer transmission, and the DFT mode is
a fixed unitary baseline.
"""
import numpy as np


def compute_precoder(cfg, H: np.ndarray) -> np.ndarray:
    mode = cfg.precoder.lower()

    if mode in ('svd', 'mrt'):

        R = np.einsum('krm,krn->mn', np.conj(H), H) / cfg.n_sc
        w, V = np.linalg.eigh((R + R.conj().T) / 2)
        order = np.argsort(w)[::-1]
        V = V[:, order]
        if mode == 'mrt':
            if cfg.n_layers != 1:
                raise ValueError('MRT precoding requires n_layers = 1.')
            return V[:, :1]
        return V[:, :cfg.n_layers]

    if mode == 'svd_persc':

        W = np.zeros((cfg.n_sc, cfg.n_tx, cfg.n_layers), dtype=complex)
        for k in range(cfg.n_sc):
            _, _, Vh = np.linalg.svd(H[k])
            W[k] = Vh.conj().T[:, :cfg.n_layers]
        return W

    if mode == 'identity':

        return np.eye(cfg.n_tx, dtype=complex)[:, :cfg.n_layers]

    if mode == 'dft':

        n = np.arange(cfg.n_tx)
        F = np.exp(-1j * 2 * np.pi * np.outer(n, n) / cfg.n_tx) / np.sqrt(cfg.n_tx)
        return F[:, :cfg.n_layers]

    raise ValueError('Unsupported precoder')
