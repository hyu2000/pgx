# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logic for dealing with coordinates.

This introduces some helpers and terminology that are used throughout Minigo.

Minigo Coordinate: This is a tuple of the form (row, column) that is indexed
    starting out at (0, 0) from the upper-left.
Flattened Coordinate: this is a number ranging from 0 - N^2 (so N^2+1
    possible values). The extra value N^2 is used to mark a 'pass' move.
SGF Coordinate: Coordinate used for SGF serialization format. Coordinates use
    two-letter pairs having the form (column, row) indexed from the upper-left
    where 0, 0 = 'aa'.
GTP Coordinate: Human-readable coordinate string indexed from bottom left, with
    the first character a capital letter for the column and the second a number
    from 1-19 for the row. Note that GTP chooses to skip the letter 'I' due to
    its similarity with 'l' (lowercase 'L').
PYGTP Coordinate: Tuple coordinate indexed starting at 1,1 from bottom-left
    in the format (column, row)

So, for a 19x19,

Coord Type      upper_left      upper_right     pass
-------------------------------------------------------
minigo coord    (0, 0)          (0, 18)         None
flat            0               18              361
SGF             'aa'            'sa'            ''
GTP             'A19'           'T19'           'pass'
"""
import itertools

import numpy as np

# We provide more than 19 entries here in case of boards larger than 19 x 19.
_SGF_COLUMNS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
_GTP_COLUMNS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'

GO_N = 5

def from_flat(flat: int):
    """Converts from a flattened coordinate to a Minigo coordinate."""
    if flat == GO_N * GO_N:
        return None
    return divmod(flat, GO_N)


def to_flat(coord: tuple) -> int:
    """Converts from a Minigo coordinate to a flattened coordinate."""
    if coord is None:
        return GO_N * GO_N
    return GO_N * coord[0] + coord[1]


def from_sgf(sgfc: str) -> tuple:
    """Converts from an SGF coordinate to a Minigo coordinate."""
    if sgfc is None or sgfc == '' or (GO_N <= 19 and sgfc == 'tt'):
        return None
    return _SGF_COLUMNS.index(sgfc[1]), _SGF_COLUMNS.index(sgfc[0])


def to_sgf(coord):
    """Converts from a Minigo coordinate to an SGF coordinate."""
    if coord is None:
        return ''
    return _SGF_COLUMNS[coord[1]] + _SGF_COLUMNS[coord[0]]


def from_gtp(gtpc):
    """Converts from a GTP coordinate to a Minigo coordinate."""
    gtpc = gtpc.upper()
    if gtpc == 'PASS':
        return None
    col = _GTP_COLUMNS.index(gtpc[0])
    row_from_bottom = int(gtpc[1:])
    return GO_N - row_from_bottom, col


def to_gtp(coord):
    """Converts from a Minigo coordinate to a GTP coordinate, e.g. E5 """
    if coord is None:
        return '--'  # 'pass'
    y, x = coord
    return '{}{}'.format(_GTP_COLUMNS[x], GO_N - y)


def flat_to_gtp(flat: int):
    return to_gtp(from_flat(flat))


def gtp_to_flat(gtpc) -> int:
    return to_flat(from_gtp(gtpc))


def locate_game_last_move(arr: np.array) -> int:
    """ batch-eval produces game records that may contain actions past actual game termination

    returns: idx_last_move + 1 (sentinel)
    """
    idx_negs = np.where(arr < 0)[0]
    idx_sentinel = len(arr)
    if len(idx_negs) > 0:
        idx_sentinel = idx_negs[0]

    # todo: find pass-pass
    return idx_sentinel


def arr_to_gtp(arr: np.array, sgf: bool = False):
    idx_termination = locate_game_last_move(arr)
    moves = arr[:idx_termination]
    if sgf:
        moves = [to_sgf(from_flat(i)) for i in moves]
        # B[cc];W[cd];
        sgf_moves = [f'{color}[{m}];' for color, m in zip(itertools.cycle(list('BW')), moves)]
        return ''.join(sgf_moves)
    return ' '.join([flat_to_gtp(i) for i in moves])