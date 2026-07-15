"""
The cross_verify_csv script performs the recorded MATLAB-Python
cross-verification. It loads the MATLAB and Python CSV result files of the
identical campaign configuration and compares every metric at every SNR
point against quantitative acceptance tolerances chosen to account for the fact
that the two implementations use independent random number streams, so
agreement is statistical rather than bit-exact. Error-rate and NMSE
metrics are compared on the logarithmic scale with a decade tolerance;
the error vector magnitude and the capacity, whose Monte Carlo variance
is small, are compared with tight relative tolerances; and the block
error rate, whose resolution at forty frames is coarse (steps of 0.025)
and whose realization variance is large, is compared with an absolute
tolerance that reflects its binomial confidence interval. The script
prints a per-point, per-metric pass/fail table and writes the complete
verification record to a text file stored beside the CSV files, so the
cross-verification is auditable and reproducible: every stored number can be traced back to
an executed run. Two structural rules protect the comparison itself: the
two files must carry exactly the same SNR grid, otherwise the script
raises instead of silently comparing an intersection, and a zero
observed bit error rate on one side only is accepted only when the other
side lies at or below the rule-of-three resolution floor of the zero
run, computed from the recorded evaluated-bit count; the floor rule
applies to the BER alone, since it is the only decade metric that is a
zero-observable event rate, while a one-sided zero NMSE fails outright.
When both sides record zero the pair is consistent by construction.
"""
import csv
import sys
import numpy as np

from config import PROJECT_ROOT


TOL = {
    'ber':  ('decade', 0.30),
    'nmse': ('decade', 0.15),
    'evmPercent': ('rel', 0.10),
    'capacityBpsHz': ('rel', 0.02),
    'bler': ('abs', 0.25),
}


REQUIRED_COLUMNS = ('snrDb',) + tuple(TOL)


def load(path):
    """Load one result CSV with publication-grade input validation:
    every required metric column must exist, every required cell must be
    non-empty, finite, and not NaN, and duplicate SNR rows are rejected
    instead of silently overwriting earlier ones."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'{path}: the CSV file contains no data rows.')
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f'{path}: missing required column(s): {missing}')
    out = {}
    for i, r in enumerate(rows, start=2):
        parsed = {}
        for k, v in r.items():
            if k in REQUIRED_COLUMNS:
                if v is None or str(v).strip() == '':
                    raise ValueError(f'{path} line {i}: empty cell in required column {k!r}.')
                value = float(v)
                if not np.isfinite(value):
                    raise ValueError(f'{path} line {i}: non-finite value {v!r} in column {k!r}.')
                parsed[k] = value
            else:
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    parsed[k] = v
        snr = parsed['snrDb']
        if snr in out:
            raise ValueError(f'{path} line {i}: duplicate SNR row {snr} dB.')
        out[snr] = parsed
    return out


def _resolution_floor(row):
    """Rule-of-three BER floor of one run: 3 / evaluated bits, when recorded."""
    n = row.get('evaluatedBits')
    return (3.0 / n) if n and n > 0 else None


def main(matlab_csv, python_csv, record_path):
    m, p = load(matlab_csv), load(python_csv)
    if set(m) != set(p):
        missing_in_python = sorted(set(m) - set(p))
        missing_in_matlab = sorted(set(p) - set(m))
        raise ValueError(
            f'SNR grids do not match. '
            f'Missing in Python: {missing_in_python}; '
            f'Missing in MATLAB: {missing_in_matlab}')
    snrs = sorted(set(m) & set(p))
    if not snrs:
        raise ValueError('Empty comparison: the CSV files share no SNR points.')
    lines = [f'Cross-verification record',
             f'MATLAB file:  {matlab_csv}', f'Python file:  {python_csv}', '']
    all_pass = True
    for snr in snrs:
        for metric, (kind, tol) in TOL.items():
            a, b = m[snr][metric], p[snr][metric]
            if kind == 'decade':
                if a == 0 and b == 0:
                    # Zero on both sides: consistent by construction,
                    # deviation undefined on the log scale.
                    ok, dev = True, float('nan')
                elif a <= 0 or b <= 0:
                    if metric == 'ber':
                        # One side observed zero errors. The pair is accepted
                        # only if the non-zero value lies at or below the
                        # rule-of-three resolution floor of the zero-side run,
                        # so a genuine discrepancy (e.g. 0 against 0.1) can no
                        # longer pass. This rule is statistically valid for a
                        # zero observed event rate only.
                        zero_row = m[snr] if a <= 0 else p[snr]
                        floor = _resolution_floor(zero_row)
                        nonzero = b if a <= 0 else a
                        if floor is not None:
                            ok, dev = (nonzero <= floor), float('inf')
                        else:
                            ok, dev = False, float('inf')
                    else:
                        # NMSE and other continuous log-scale metrics are not
                        # event rates: a zero on one side only is a genuine
                        # disagreement, not a small-sample artifact.
                        ok, dev = False, float('inf')
                else:
                    dev = abs(np.log10(a) - np.log10(b)); ok = dev <= tol
                unit = 'decades'
            elif kind == 'rel':
                ref = max(abs(a), 1e-12)
                dev = abs(a - b) / ref; ok = dev <= tol; unit = 'relative'
            else:
                dev = abs(a - b); ok = dev <= tol; unit = 'absolute'
            all_pass &= ok
            lines.append(f'{snr:5.0f} dB  {metric:<16} MATLAB {a:12.6g}  '
                         f'Python {b:12.6g}  dev {dev:8.4f} {unit:<8} '
                         f'tol {tol:<5}  {"PASS" if ok else "FAIL"}')
    lines.append('')
    lines.append('OVERALL: ' + ('PASS - the Python and MATLAB implementations agree '
                 'within the acceptance tolerances at every SNR point'
                 if all_pass else 'FAIL - at least one comparison exceeded tolerance'))
    record = '\n'.join(lines)
    print(record)
    with open(record_path, 'w') as f:
        f.write(record + '\n')
    print(f'\nSaved cross-verification record: {record_path}')
    return bool(all_pass)


if __name__ == '__main__':
    ok = main(sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / 'results/matlab_compact_ref.csv'),
              sys.argv[2] if len(sys.argv) > 2 else str(PROJECT_ROOT / 'results/python_compact_mainflow.csv'),
              sys.argv[3] if len(sys.argv) > 3 else str(PROJECT_ROOT / 'results/cross_verification_record.txt'))
    sys.exit(0 if ok else 1)
