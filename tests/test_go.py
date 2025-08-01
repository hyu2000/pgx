from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from pgx._src.games.go import _count_ji, _count_scores
from pgx.go import Go, State
from pgx.experimental.go import from_sgf


# only for debug
def _show(state: State) -> None:
    BLACK_CHAR = "@"
    WHITE_CHAR = "O"
    POINT_CHAR = "+"
    print("===========")
    board_size = BOARD_SIZE
    for xy in range(board_size * board_size):
        if state._x.board[xy] > 0:
            print(" " + BLACK_CHAR, end="")
        elif state._x.board[xy] < 0:
            print(" " + WHITE_CHAR, end="")
        else:
            print(" " + POINT_CHAR, end="")

        if xy % board_size == board_size - 1:
            print()


BOARD_SIZE = 5
env = Go(size=BOARD_SIZE)
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def test_init():
    """ game can start with either player 0 or 1 (randomized) """
    for i in range(10):
        key = jax.random.PRNGKey(i)
        state = init(key=key)
        # underlying game color: 0 == black
        assert state._x.color == 0
        print(f'iter {i} {state._step_count} {state.current_player}')
        if state.current_player == 1:
            continue


def test_no_jit():
    env = Go(size=5, komi=0.5)
    key = jax.random.PRNGKey(0)
    state0 = env.init(key)
    assert state0.current_player == 0
    assert state0._x.color == 0
    state1 = env.step(state0, action=1)
    assert state1.current_player == 1
    assert state1._x.color == 1
    state2 = env.step(state1, action=2)
    assert state2.current_player == 0
    assert state2._x.color == 0

    key = jax.random.PRNGKey(1)
    state10 = env.init(key)
    assert state10.current_player == 1  # would this result in different observation?
    assert state10._x.color == 0
    state11 = env.step(state10, action=1)
    assert state11.current_player == 0
    assert state11._x.color == 1
    state12 = env.step(state11, action=2)
    assert state12.current_player == 1
    assert state12._x.color == 0

    # seems ok
    obs0 = state0.observation
    obs10 = state10.observation
    assert (obs0 == obs10).all()
    assert (state1.observation == state11.observation).all()


def test_go5C2():
    key = jax.random.PRNGKey(0)

    env = Go(size=5, komi=0.5)
    state0 = env.init(key=key)
    state1 = env.step(state0, 17)
    _show(state1)
    print(state1.observation.shape)
    obs1 = state1.observation.astype(jnp.int8)
    # obs1 = jnp.moveaxis(obs1, -1, 0)
    print(obs1[:, :, -1])  # white to move
    print(obs1[:, :, 0])   # current board, my color (white)
    print(obs1[:, :, 1])   # current board, opp color (black)
    assert(state1._step_count == 1)


def test_go5C2env():
    key = jax.random.PRNGKey(0)

    env = Go(size=5, komi=0.5, open_move=17)
    state0 = env.init(key=key)
    _show(state0)
    obs0 = state0.observation.astype(jnp.int8)
    print(obs0[:, :, -1])
    print(obs0[:, :,  0])
    print(obs0[:, :,  1])
    state1 = env.step(state0, 12)  # C3
    _show(state1)
    assert(state1._step_count == 1)
    obs1 = state1.observation.astype(jnp.int8)
    print(obs1.shape, obs1.dtype)
    print(obs1[:, :, -1])  # black to move
    print(obs1[:, :, 0])   # black's stones
    print(obs1[:, :, 1])   # white's stones


def test_C2jit():
    """ see if go5C2 env is safe under jit """
    key = jax.random.PRNGKey(0)

    env = Go(size=5, komi=0.5, open_move=17)
    init = jax.jit(env.init)
    step = jax.jit(env.step)

    state0 = init(key=key)
    _show(state0)
    state1 = step(state0, 12)
    _show(state1)
    obs1 = state1.observation.astype(jnp.int8)
    print(obs1.shape, obs1.dtype)
    print(obs1[:, :, -1])  # black to move
    print(obs1[:, :, 0])   # black's stones
    print(obs1[:, :, 1])   # white's stones


def test_end_by_pass():
    key = jax.random.PRNGKey(0)

    state = init(key=key)
    state = step(state=state, action=25)
    assert state._x.consecutive_pass_count == 1
    assert not state.terminated
    state = step(state=state, action=0)
    assert state._x.consecutive_pass_count == 0
    assert not state.terminated
    state = step(state=state, action=25)
    assert state._x.consecutive_pass_count == 1
    assert not state.terminated
    state = step(state=state, action=25)
    assert state._x.consecutive_pass_count == 2
    assert state.terminated


def test_step():
    """
    https://www.cosumi.net/replay/?b=You&w=COSUMI&k=0&r=0&bs=5&gr=ccbccdcbdbbadabdbecaacabecaddeaettceedbetttt
    """
    key = jax.random.PRNGKey(1)
    state = init(key=key)
    assert state.current_player == 1

    state = step(state=state, action=12)  # BLACK
    state = step(state=state, action=11)  # WHITE
    state = step(state=state, action=17)
    state = step(state=state, action=7)
    state = step(state=state, action=8)
    state = step(state=state, action=1)
    state = step(state=state, action=3)
    state = step(state=state, action=16)
    state = step(state=state, action=21)
    state = step(state=state, action=2)
    state = step(state=state, action=10)
    state = step(state=state, action=5)
    state = step(state=state, action=14)
    state = step(state=state, action=15)
    state = step(state=state, action=23)
    state = step(state=state, action=20)
    state = step(state=state, action=25)  # pass
    state = step(state=state, action=22)
    state = step(state=state, action=19)
    state = step(state=state, action=21)
    state = step(state=state, action=25)  # pass
    state = step(state=state, action=25)  # pass

    expected_board: jnp.ndarray = jnp.array(
        [
            [ 0, -1, -1, 1, 0],
            [-1,  0, -1, 1, 0],
            [ 0, -1,  1, 0, 1],
            [-1, -1,  1, 0, 1],
            [-1, -1, -1, 1, 0],
        ]
    )  # type:ignore
    """
      [ 0 1 2 3 4 ]
    [0] + O O @ +
    [1] O + O @ +
    [2] + O @ + @
    [3] O O @ + @
    [4] O O O @ +
    """
    assert (jnp.clip(state._x.board, -1, 1) == expected_board.ravel()).all()
    assert state.terminated

    # 同点なのでコミの分 黒 == player_1 の負け
    assert (state.rewards == jnp.array([1, -1])).all()


def test_from_sgf():
    state = from_sgf("(;GM[1]FF[4]DT[2023-06-22 19 51 52]SZ[9]KM[6.5]RU[japanese]PB[AI (KataGo)ㅤ​]BR[9d]PW[AI (KataGo)ㅤ​]WR[9d]RE[B+1.5]CA[UTF-8]AP[KaTrain:1.13.0]KTV[1.0]C[SGF generated by KaTrain 1.13.0ㅤ​];B[fe];W[de];B[ec];W[cc];B[cg];W[gc];B[ef];W[fd];B[ed];W[ee];B[ge];W[ff];B[eg];W[fc];B[he];W[cf];B[fb];W[gb];B[eb];W[db];B[bf];W[ce];B[hc];W[hb];B[ic];W[bg];B[bh];W[dg];B[ch];W[gg];B[ib];W[eh];B[fg];W[fh];B[gf];W[dh];B[ag];W[hg];B[ac];W[df];B[ff];W[ab];B[ae];W[bc];B[ad];W[ie];B[if];W[ig];B[id];W[hf];B[ga];W[cb];B[di];W[be];B[ha];W[ei];B[ci];W[ea];B[dd];W[ie];B[bb];W[ba];B[if];W[ah];B[ai];W[ie];B[da];W[ca];B[if];W[da];B[ie];W[cd];B[gd];W[fa];B[af];W[bd];B[dc];W[];B[])")
    state.save_svg("tests/assets/go/from_sgf_001.svg")
    expected_board: jnp.ndarray = jnp.array(
        [
            [ 0, -1, -1, -1, -1, -1,  1,  1,  0],
            [-1,  0, -1, -1,  1,  1,  0,  0,  1],
            [ 1, -1, -1,  1,  1,  0,  0,  1,  1],
            [ 1, -1, -1,  1,  1,  0,  1,  0,  1],
            [ 1, -1, -1, -1, -1,  1,  1,  1,  1],
            [ 1,  1, -1, -1,  1,  1,  1, -1,  1],
            [ 1,  0,  1, -1,  1,  1, -1, -1, -1],
            [ 0,  1,  1, -1, -1, -1,  0,  0,  0],
            [ 1,  0,  1,  1, -1,  0,  0,  0,  0],
        ]
    )  # type:ignore
    assert (jnp.clip(state._x.board, -1, 1) == expected_board.ravel()).all()
    assert state.terminated


    # 検討譜は読み込めない
    state = from_sgf("(;FF[4]GM[1]CA[UTF-8]AP[besogo:0.0.0-alpha]SZ[9]ST[0];B[fe];W[de];B[ec];W[fg](;B[ef];W[eg];B[df];W[cc];B[gg];W[gf];B[hg];W[hf];B[ge];W[ff];B[he];W[if];B[dg];W[ce];B[ee];W[bg];B[db];W[cb];B[ca];W[ba];B[da];W[ab];B[dh];W[eh];B[ei];W[fi];B[di];W[gh];B[ie];W[hh];B[cf];W[bf];B[bh];W[cg];B[dd];W[cd];B[ch];W[dc];B[eb];W[ed];B[fd];W[ah];B[bi];W[ag];B[ai];W[be];B[dd];W[];B[ed];W[])(;B[cc];W[hf];B[ef];W[];B[]))")
    state.save_svg("tests/assets/go/from_sgf_002.svg")
    # 分岐先は終局しているが主分岐は終局していない
    assert not state.terminated

    # 初手からの分岐
    state = from_sgf("(;FF[4]GM[1]CA[UTF-8]AP[besogo:0.0.0-alpha]SZ[9]ST[0](;B[ee])(;B[eg])(;B[ec]))")
    state.save_svg("tests/assets/go/from_sgf_003.svg")
    board = jnp.clip(state._x.board, -1, 1)
    assert board[40] == 1
    assert not state.terminated

    # 19×19
    state = from_sgf("(;AP[GOWrite:2.3.48]GM[1]CA[UTF-8]SZ[19]FF[4]ST[2]PB[FanHui]BR[二段]PW[AlphaGo]WR[Google]KM[3又3/4]RE[W+1又1/4子]EV[FanHui_AlphaGo対戦]RO[1]DT[2015-10-05]OT[3x30 Byo-yomi]TM[60]SO[新浪囲棋]PC[Google,London];B[pd];W[dd];B[pp];W[dq];B[co];W[dl];B[dn];W[fp];B[bq];W[jq];B[cf];W[ch];B[fd];W[df];B[dg];W[cg];B[dc];W[ce];B[cc];W[hc];B[fb];W[nc];B[qf];W[pb];B[bf];W[be];B[ef];W[de];B[qc];W[kc];B[qn];W[cm];B[cr];W[mq];B[oq];W[qm];B[pm];W[ql];B[rn];W[pl];B[om];W[qi];B[hq];W[hp];B[gq];W[gp];B[iq];W[ip];B[jr];W[kq];B[er];W[rg];B[qg];W[rf];B[re];W[qh];B[kr];W[ln];B[sf];W[rh];B[qb];W[fe];B[el];W[ek];B[fk];W[ej];B[fl];W[dm];B[fj];W[il];B[fi];W[ei];B[fh];W[gd];B[dh];W[ci];B[di];W[cj];B[fn];W[em];B[hn];W[in];B[en];W[oo];B[np];W[nn];B[po];W[hm];B[fm];W[nl];B[og];W[nr];B[or];W[nh];B[oj];W[ol];B[oh];W[ni];B[oi];W[ng];B[nf];W[mf];B[ne];W[me];B[gc];W[hb];B[bd];W[ed];B[fc];W[ff];B[ae];W[bg];B[af];W[eh];B[fg];W[eg];B[ge];W[hd];B[gf];W[ee];B[if];W[fa];B[ga];W[gb];B[ea];W[ec];B[eb];W[nd];B[je];W[pe];B[oe];W[od];B[of];W[pc];B[qd];W[jh];B[kd];W[lc];B[nj];W[hh];B[hg];W[mj];B[mk];W[lj];B[nk];W[lk];B[pa];W[rm];B[mp];W[lp];B[lr];W[lq];B[bn];W[qq];B[rp];W[rq];B[qp];W[ag];B[ad];W[cp];B[bp];W[oa];B[qa];W[dr];B[ds];W[ob];B[ml];W[nm];B[pn];W[hj];B[kg];W[jg];B[kf];W[kh];B[bl];W[bm];B[am];W[bk];B[jc];W[jb];B[lm];W[km];B[mn];W[mo];B[mm];W[no];B[kl];W[jm];B[ll];W[lg];B[jk];W[cl];B[qj];W[rj];B[gm];W[ho];B[al];W[ak];B[an];W[ic];B[mr];W[nq];B[ns];W[op];B[pq];W[jd];B[pj];W[sg];B[ii];W[se];B[sd];W[ih];B[ji];W[hi];B[ie];W[ld];B[ke];W[he];B[gg];W[eq];B[fq];W[ep];B[cq];W[gn];B[ki];W[li];B[ik];W[sn];B[so];W[sm];B[dp];W[eo];B[id];W[jc];B[do];W[fo];B[hk];W[hl];B[cb];W[ph];B[pg];W[qk];B[ha];W[ia];B[fa];W[ca];B[ba];W[dj];B[cd];W[sf];B[gl];W[gj];B[gk];W[ij];B[kk];W[lh];B[ig];W[lf];B[le];W[gh];B[kj];W[jf];B[hf];W[jl];B[jj];W[gi];B[pi];W[cn];B[pk];W[ok];B[on]C[271手白1又1/4子勝])")
    state.save_svg("tests/assets/go/from_sgf_004.svg")
    # ダウンロードした棋譜がpassまで書いていないためterminatedになっていないが、有効手はすべて打ち終わっている
    assert not state.terminated

    state = from_sgf("(;EV[第40期棋聖戦Bリーグ1組]RO[7回戦]DT[2015/09/03]KM[6.5]PB[瀬戸大樹]PW[黄翊祖]BR[八段]WR[七段]PC[関西棋院]RE[W+R]GC[186手完白中押し勝ち];B[pd];W[dp];B[qp];W[dc];B[de];W[op];B[fd];W[pn];B[qn];W[pm];B[qm];W[pl];B[eb];W[qq];B[rq];W[pq];B[rr];W[jp];B[cn];W[eo];B[dl];W[ce];B[cf];W[df];B[dd];W[cd];B[bf];W[ec];B[cc];W[cb];B[bc];W[fc];B[fe];W[gd];B[ge];W[hd];B[he];W[id];B[hq];W[co];B[er];W[bn];B[cq];W[cm];B[nc];W[rd];B[qd];W[re];B[rc];W[rb];B[qc];W[qg];B[jc];W[ie];B[if];W[ke];B[jf];W[kd];B[kc];W[kh];B[cl];W[em];B[lp];W[iq];B[hp];W[ko];B[mn];W[lm];B[nq];W[qo];B[ro];W[po];B[or];W[pr];B[kq];W[ir];B[mr];W[rl];B[rm];W[ql];B[sl];W[sk];B[sm];W[fq];B[fr];W[gr];B[hr];W[hs];B[gq];W[gs];B[fk];W[gl];B[gk];W[hk];B[hj];W[ij];B[fl];W[gm];B[ik];W[hl];B[hi];W[ii];B[ih];W[jh];B[lg];W[lh];B[mh];W[mg];B[fm];W[fn];B[gn];W[hn];B[go];W[ho];B[il];W[hm];B[kj];W[hh];B[ig];W[gj];B[gi];W[fj];B[ej];W[fi];B[gh];W[di];B[cj];W[dj];B[dk];W[ci];B[bi];W[bj];B[bk];W[aj];B[bh];W[ck];B[bm];W[dm];B[cj];W[ji];B[ck];W[bb];B[ng];W[mf];B[bd];W[sn];B[rn];W[am];B[bl];W[mi];B[mm];W[ml];B[ln];W[kn];B[jr];W[an];B[ai];W[dq];B[dr];W[bq];B[br];W[cp];B[cr];W[rj];B[nh];W[ni];B[nf];W[me];B[ph];W[mc];B[mb];W[nd];B[lc];W[oe];B[ll];W[km];B[nl];W[mk];B[nk];W[md];B[oc];W[mj];B[oi];W[oj];B[nj];W[li])")
    state.save_svg("tests/assets/go/from_sgf_005.svg")
    # 中押し勝（降参）
    assert not state.terminated

    state = from_sgf("(;CA[shift_jis]SZ[19]AP[MultiGo:4.4.4]GN[Google Challenge Match #4]EV[Google DeepMind Challenge Match 第4局]DT[2016-03-13]PC[韓国]PB[AlphaGo]BR[1p]BT[Google]PW[李世ドル]WR[9p]WT[韓国]KM[7.5]TM[2ｈ]RE[W+R]MULTIGOGM[1];B[pd];W[dp];B[cd];W[qp];B[op];W[oq];B[nq];W[pq];B[cn];W[fq];B[mp];W[po];B[iq];W[ec];B[hd];W[cg];B[ed];W[cj];B[dc];W[bp];B[nc];W[qi];B[ep];W[eo];B[dk];W[fp];B[ck];W[dj];B[ej];W[ei];B[fi];W[eh];B[fh];W[bj];B[fk];W[fg];B[gg];W[ff];B[gf];W[mc];B[md];W[lc];B[nb];W[id];B[hc];W[jg];B[pj];W[pi];B[oj];W[oi];B[ni];W[nh];B[mh];W[ng];B[mg];W[mi];B[nj];W[mf];B[li];W[ne];B[nd];W[mj];B[lf];W[mk];B[me];W[nf];B[lh];W[qj];B[kk];W[ik];B[ji];W[gh];B[hj];W[ge];B[he];W[fd];B[fc];W[ki];B[jj];W[lj];B[kh];W[jh];B[ml];W[nk];B[ol];W[ok];B[pk];W[pl];B[qk];W[nl];B[kj];W[ii];B[rk];W[om];B[pg];W[ql];B[cp];W[co];B[oe];W[rl];B[sk];W[rj];B[hg];W[ij];B[km];W[gi];B[fj];W[jl];B[kl];W[gl];B[fl];W[gm];B[ch];W[ee];B[eb];W[bg];B[dg];W[eg];B[en];W[fo];B[df];W[dh];B[im];W[hk];B[bn];W[if];B[gd];W[fe];B[hf];W[ih];B[bh];W[ci];B[ho];W[go];B[or];W[rg];B[dn];W[cq];B[pr];W[qr];B[rf];W[qg];B[qf];W[jc];B[gr];W[sf];B[se];W[sg];B[rd];W[bl];B[bk];W[ak];B[cl];W[hn];B[in];W[hp];B[fr];W[er];B[es];W[ds];B[ah];W[ai];B[kd];W[ie];B[kc];W[kb];B[gk];W[ib];B[qh];W[rh];B[qs];W[rs];B[oh];W[sl];B[of];W[sj];B[ni];W[nj];B[oo];W[jp]N[B resigned])")
    state.save_svg("tests/assets/go/from_sgf_006.svg")
    # 中押し勝（降参）
    assert not state.terminated

    # 分岐あり
    state = from_sgf("(;FF[4]GM[1]CA[UTF-8]AP[besogo:0.0.0-alpha]SZ[19]ST[0];B[pd];W[qf];B[nc](;W[rd];B[qc];W[qi])(;W[qd];B[qc];W[rc];B[qe];W[rd];B[pf];W[re];B[pe];W[qg]))")
    state.save_svg("tests/assets/go/from_sgf_007.svg")
    board = jnp.clip(state._x.board, -1, 1)
    assert board[168] == -1
    assert board[55] == 0

    # from_sgf_006と全く同じだが分岐がある
    state = from_sgf("(;CA[shift_jis]SZ[19]AP[MultiGo:4.4.4]GN[Google Challenge Match #4]EV[Google DeepMind Challenge Match 第4局]DT[2016-03-13]PC[韓国]PB[AlphaGo]BR[1p]BT[Google]PW[李世ドル]WR[9p]WT[韓国]KM[7.5]TM[2ｈ]RE[W+R]MULTIGOGM[1];B[pd];W[dp];B[cd];W[qp];B[op];W[oq];B[nq];W[pq];B[cn];W[fq];B[mp];W[po];B[iq];W[ec];B[hd];W[cg];B[ed];W[cj];B[dc];W[bp];B[nc];W[qi];B[ep];W[eo];B[dk];W[fp];B[ck];W[dj];B[ej](;W[ei];B[fi];W[eh];B[fh];W[bj];B[fk];W[fg];B[gg];W[ff];B[gf];W[mc];B[md];W[lc];B[nb];W[id];B[hc];W[jg];B[pj];W[pi];B[oj];W[oi];B[ni];W[nh];B[mh];W[ng];B[mg];W[mi];B[nj];W[mf];B[li];W[ne];B[nd];W[mj];B[lf];W[mk];B[me];W[nf];B[lh];W[qj];B[kk];W[ik];B[ji];W[gh];B[hj];W[ge];B[he];W[fd];B[fc];W[ki];B[jj];W[lj];B[kh];W[jh];B[ml];W[nk];B[ol];W[ok];B[pk];W[pl];B[qk];W[nl];B[kj];W[ii];B[rk];W[om];B[pg];W[ql];B[cp];W[co];B[oe];W[rl];B[sk];W[rj];B[hg];W[ij];B[km];W[gi];B[fj];W[jl];B[kl];W[gl];B[fl];W[gm];B[ch];W[ee];B[eb];W[bg];B[dg];W[eg];B[en];W[fo];B[df];W[dh];B[im];W[hk];B[bn];W[if];B[gd];W[fe];B[hf];W[ih];B[bh];W[ci];B[ho];W[go];B[or];W[rg];B[dn];W[cq];B[pr];W[qr];B[rf];W[qg];B[qf];W[jc];B[gr];W[sf];B[se];W[sg];B[rd];W[bl];B[bk];W[ak];B[cl];W[hn];B[in];W[hp];B[fr];W[er];B[es];W[ds];B[ah];W[ai];B[kd];W[ie];B[kc];W[kb];B[gk];W[ib];B[qh];W[rh];B[qs];W[rs];B[oh];W[sl];B[of];W[sj];B[ni];W[nj];B[oo];W[jp]N[B resigned])(;W[ek];B[bj];W[bi];B[bk];W[dh];B[fj];W[el];B[en];W[fn];B[em];W[fm];B[dl];W[gk];B[fh]))")
    state.save_svg("tests/assets/go/from_sgf_008.svg")
    assert not state.terminated




def test_ko():
    key = jax.random.PRNGKey(0)

    state: State = init(key=key)
    assert state.current_player == 1
    state = step(state=state, action=2)  # BLACK
    state = step(state=state, action=17)  # WHITE
    state = step(state=state, action=6)  # BLACK
    state = step(state=state, action=13)  # WHITE
    state = step(state=state, action=8)  # BLACK
    state = step(state=state, action=11)  # WHITE
    state = step(state=state, action=12)  # BLACK
    state = step(state=state, action=7)  # WHITE

    """
    ===========
    + + @ + +
    + @ + @ +
    + O @ O +
    + + O + +
    + + + + +
    ===========
    + + @ + +
    + @ O @ +
    + O + O +
    + + O + +
    + + + + +
    """
    assert state._x.ko == 12

    loser = state.current_player
    state1: State = step(
        state=state, action=12
    )  # BLACK
    # ルール違反により黒 = player_id=1 の負け
    assert state1.terminated
    assert state1.rewards[loser] == -1.
    assert state1.rewards.sum() == 0.

    state2: State = step(state=state, action=0)  # BLACK
    # 回避した場合
    assert not state2.terminated
    assert state2._x.ko == -1

    # see #468
    state: State = init(key=key)
    state = step(state, action=2)
    state = step(state, action=9)
    state = step(state, action=18)
    state = step(state, action=5)
    state = step(state, action=11)
    state = step(state, action=22)
    state = step(state, action=8)
    state = step(state, action=14)
    state = step(state, action=25)
    state = step(state, action=1)
    state = step(state, action=24)
    state = step(state, action=23)
    state = step(state, action=7)
    state = step(state, action=4)
    state = step(state, action=16)
    state = step(state, action=15)
    state = step(state, action=19)
    state = step(state, action=6)
    state = step(state, action=20)
    state = step(state, action=25)
    state = step(state, action=12)
    state = step(state, action=3)
    state = step(state, action=21)
    state = step(state, action=10)
    state = step(state, action=17)
    state = step(state, action=25)
    state = step(state, action=13)
    state = step(state, action=4)
    state = step(state, action=14)
    state = step(state, action=23)
    state = step(state, action=0)
    assert state._x.ko == -1

    # see #468
    state: State = init(key=key)
    state = step(state, action=1)
    state = step(state, action=16)
    state = step(state, action=9)
    state = step(state, action=11)
    state = step(state, action=14)
    state = step(state, action=6)
    state = step(state, action=24)
    state = step(state, action=4)
    state = step(state, action=25)
    state = step(state, action=17)
    state = step(state, action=21)
    state = step(state, action=18)
    state = step(state, action=13)
    state = step(state, action=23)
    state = step(state, action=8)
    state = step(state, action=0)
    state = step(state, action=5)
    state = step(state, action=25)
    state = step(state, action=15)
    state = step(state, action=19)
    state = step(state, action=22)
    state = step(state, action=25)
    state = step(state, action=12)
    state = step(state, action=10)
    state = step(state, action=2)
    state = step(state, action=25)
    state = step(state, action=3)
    state = step(state, action=20)
    assert state._x.ko == -1

    # Ko after pass
    state: State = init(key=key)
    state = step(state, action=17)
    state = step(state, action=25)
    state = step(state, action=20)
    state = step(state, action=24)
    state = step(state, action=6)
    state = step(state, action=13)
    state = step(state, action=12)
    state = step(state, action=18)
    state = step(state, action=22)
    state = step(state, action=5)
    state = step(state, action=8)
    state = step(state, action=10)
    state = step(state, action=11)
    state = step(state, action=14)
    state = step(state, action=2)
    state = step(state, action=9)
    state = step(state, action=1)
    state = step(state, action=23)
    state = step(state, action=16)
    state = step(state, action=4)
    state = step(state, action=0)
    state = step(state, action=19)
    state = step(state, action=15)
    state = step(state, action=25)
    state = step(state, action=21)
    state = step(state, action=25)
    state = step(state, action=5)
    state = step(state, action=25)
    state = step(state, action=3)
    state = step(state, action=14)
    state = step(state, action=4)
    state = step(state, action=19)
    state = step(state, action=7)
    state = step(state, action=23)
    state = step(state, action=18)
    state = step(state, action=13)
    state = step(state, action=24)
    state = step(state, action=25)  # pass
    assert state._x.ko == -1

    # see #479
    actions = [107, 11, 56, 41, 300, 19, 228, 231, 344, 257, 35, 32, 57, 276, 0, 277, 164, 15, 187, 179, 357, 255, 150, 211, 256,
     190, 297, 303, 358, 189, 322, 3, 129, 64, 13, 336, 22, 286, 264, 192, 55, 360, 23, 31, 113, 119, 195, 98, 208, 294,
     240, 241, 149, 280, 118, 296, 245, 99, 335, 226, 29, 287, 84, 248, 225, 351, 202, 20, 137, 274, 232, 85, 36, 141,
     108, 95, 282, 93, 337, 216, 58, 131, 283, 10, 106, 243, 318, 220, 136, 34, 127, 293, 80, 165, 125, 83, 114, 105,
     30, 61, 147, 71, 109, 173, 87, 233, 76, 361, 66, 115, 212, 200, 346, 197, 54, 326, 298, 167, 347, 4, 354, 16, 140,
     144, 68, 178, 24, 204, 285, 203, 316, 307, 146, 37, 201, 268, 176, 133, 25, 227, 310, 291, 132, 352, 123, 184, 343,
     299, 90, 267, 334, 134, 7, 110, 321, 182, 281, 92, 222, 96, 329, 70, 340, 207, 323, 138, 308, 100, 49, 78, 5, 126,
     317, 17, 349, 160, 261, 266, 306, 221, 355, 327, 324, 284, 236, 60, 359, 174, 252, 46, 260, 114, 163, 235, 250,
     206, 239, 2, 166, 328, 128, 104, 341, 224, 74, 198, 304, 295, 101, 88, 360, 325, 199, 38, 263, 270, 151, 331, 230,
     33, 152, 48, 47, 28, 122, 161, 273, 103, 143, 238, 121, 52, 333, 244, 218, 265, 361, 77, 275, 185, 172, 350, 194,
     59, 53, 21, 272, 319, 320, 158, 251, 253, 135, 27, 196, 180, 188, 345, 254, 130, 42, 156, 259, 332, 361, 18, 82,
     86, 191, 249, 51, 45, 348, 217, 63, 302, 292, 155, 313, 205, 6, 237, 279, 229, 258, 234, 262, 40, 73, 142, 219,
     330, 111, 186, 153, 311, 336, 44, 12, 62, 215, 39, 299, 9, 269, 275, 157, 225, 361, 177, 361, 162, 81, 76, 183,
     168, 247, 309, 145, 210, 221, 65, 301, 1, 289, 120, 315, 353, 305, 67, 214, 79, 314, 290, 47, 181, 346, 175, 89,
     312, 43, 231, 329, 102, 91, 208, 139, 236, 348, 66, 26, 8, 94, 169, 271, 339, 58, 69, 80, 349, 170, 23, 159, 347,
     288, 154, 270, 6, 187, 22, 42, 148, 193, 346, 126, 116, 242, 124, 159, 14, 12, 144, 26, 24, 361, 223, 7, 361, 63,
     117, 112, 5, 81, 118, 135, 82, 92, 140, 123, 97, 278, 47, 361, 137, 230, 220]
    env = Go(size=19)
    env.init = jax.jit(env.init)
    env.step = jax.jit(env.step)
    state = env.init(jax.random.PRNGKey(0))
    for a in actions:
        state = env.step(state, a)
    assert state._x.ko == -1
    assert state.legal_action_mask[231]

def test_observe():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state.current_player == 1
    # player 0 is white, player 1 is black
    obs = observe(state, 1)   # black turn, black view
    assert (obs[:, :, -1] == 0).all()
    obs = observe(state, 0)   # black turn, white view
    assert (obs[:, :, -1] == 1).all()

    state = step(state=state, action=0)
    state = step(state=state, action=1)
    state = step(state=state, action=2)
    state = step(state=state, action=3)
    state = step(state=state, action=4)
    state = step(state=state, action=5)
    state = step(state=state, action=6)
    state = step(state=state, action=7)
    # ===========
    # + O + O @
    # O @ O + +
    # + + + + +
    # + + + + +
    # + + + + +
    # fmt: off
    curr_board = jnp.int32(
        [[ 0, -1,  0, -1, 1],
         [-1,  1, -1,  0, 0],
         [ 0,  0,  0,  0, 0],
         [ 0,  0,  0,  0, 0],
         [ 0,  0,  0,  0, 0]]
    )
    # fmt: on
    assert state.current_player == 1
    assert state._x.color % 2 == 0  # black turn
    obs = observe(state, 0)   # white
    assert obs.shape == (5, 5, 17)
    assert (obs[:, :, 0] == (curr_board == -1)).all()
    assert (obs[:, :, 1] == (curr_board == 1)).all()
    assert (obs[:, :, -1] == 1).all()

    obs = observe(state, 1)  # black
    assert obs.shape == (5, 5, 17)
    assert (obs[:, :, 0] == (curr_board == 1)).all()
    assert (obs[:, :, 1] == (curr_board == -1)).all()
    assert (obs[:, :, -1] == 0).all()


def test_legal_action():
    key = jax.random.PRNGKey(0)

    # =====
    # @ + @ + @
    # + @ + @ +
    # @ + @ + @
    # + + + + +
    # + + + + +
    # fmt:off
    expected = jnp.array([
        False, False, False, False, False,
        False, False, False, False, False,
        False, True, False, True, False,
        True, True, True, True, True,
        True, True, True, True, True, True])
    # fmt:on
    state = init(key=key)
    state = step(state=state, action=0)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=4)
    state = step(state=state, action=25)
    state = step(state=state, action=6)
    state = step(state=state, action=25)
    state = step(state=state, action=8)
    state = step(state=state, action=25)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=14)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # =====
    # + @ @ @ +
    # @ O + O @
    # + @ @ @ +
    # + + + + +
    # + + + + +
    # fmt:off
    expected = jnp.array([
        False, False, False, False, False,
        False, False, False, False, False,
        True, False, False, False, True,
        True, True, True, True, True,
        True, True, True, True, True, True])
    # fmt:on
    # white 8
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=3)
    state = step(state=state, action=25)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    state = step(state=state, action=9)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=25)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # black 13
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=6)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=8)
    state = step(state=state, action=3)
    state = step(state=state, action=25)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    state = step(state=state, action=9)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=13)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # black 9
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=6)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=8)
    state = step(state=state, action=3)
    state = step(state=state, action=25)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    state = step(state=state, action=9)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=13)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # =====
    # + + O + +
    # + O @ O +
    # O @ + @ O
    # + O @ O +
    # + + O + +
    # fmt:off
    expected_b = jnp.array([
        True, True, False, True, True,
        True, False, False, False, True,
        False, False, False, False, False,
        True, False, False, False, True,
        True, True, False, True, True, True])
    expected_w = jnp.array([
        True, True, False, True, True,
        True, False, False, False, True,
        False, False, True, False, False,
        True, False, False, False, True,
        True, True, False, True, True, True])
    # fmt:on
    state = init(key=key)
    state = step(state=state, action=7)  # BLACK
    state = step(state=state, action=2)  # WHITE
    state = step(state=state, action=11)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=17)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=16)
    state = step(state=state, action=25)
    state = step(state=state, action=18)
    state = step(state=state, action=25)
    state = step(state=state, action=22)  # WHITE
    assert jnp.all(state.legal_action_mask == expected_b)
    state = step(state=state, action=25)  # BLACK
    assert jnp.all(state.legal_action_mask == expected_w)

    # =====
    # + @ @ @ +
    # @ O @ + @
    # @ O @ O @
    # @ O @ O @
    # @ O O O @
    # fmt:off
    # black 24
    expected_w1 = jnp.array([
        True, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False, True])
    # white pass
    expected_b = jnp.array([
        True, False, False, False, True,
        False, False, False, True, False,
        False, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False, True])
    # black 8
    expected_w2 = jnp.array([
        False, False, False, False, False,
        False, True, False, False, False,
        False, True, False, True, False,
        False, True, False, True, False,
        False, True, True, True, False, True])
    # fmt:on
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=6)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=11)
    state = step(state=state, action=3)
    state = step(state=state, action=13)
    state = step(state=state, action=5)
    state = step(state=state, action=16)
    state = step(state=state, action=7)
    state = step(state=state, action=18)
    state = step(state=state, action=9)
    state = step(state=state, action=21)
    state = step(state=state, action=10)
    state = step(state=state, action=22)
    state = step(state=state, action=12)
    state = step(state=state, action=23)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=24)
    state = step(state=state, action=25)
    state = step(state=state, action=20)
    assert jnp.all(state.legal_action_mask == expected_w1)
    state = step(state=state, action=25)
    assert jnp.all(state.legal_action_mask == expected_b)
    state = step(state=state, action=8)
    assert jnp.all(state.legal_action_mask == expected_w2)

    # =====
    # random
    env = Go(size=BOARD_SIZE)
    env.init = jax.jit(env.init)
    env.step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)
    state = env.init(key=key)
    for _ in range(50):  # 5 * 5 * 2 = 50
        assert np.where(state.legal_action_mask)[0][-1]

        legal_actions = np.where(state.legal_action_mask)[0][:-1]
        illegal_actions = np.where(~state.legal_action_mask)[0][:-1]
        for action in legal_actions:
            _state = env.step(state=state, action=action)
            if _state._step_count < 50:
                assert not _state.terminated
            else:
                assert _state.terminated
        for action in illegal_actions:
            _state = env.step(state=state, action=action)
            assert _state.terminated
        if len(legal_actions) == 0:
            a = BOARD_SIZE * BOARD_SIZE
        else:
            key = jax.random.PRNGKey(0)
            key, subkey = jax.random.split(key)
            a = jax.random.choice(subkey, legal_actions)
        state = env.step(state=state, action=a)


def test_counting_ji():
    key = jax.random.PRNGKey(0)
    count_ji = jax.jit(_count_ji, static_argnums=(2,))
    BLACK, WHITE = 1, -1

    # =====
    # @ + @ + @
    # + @ + @ +
    # @ + @ + @
    # + + + + +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=0)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=4)
    state = step(state=state, action=25)
    state = step(state=state, action=6)
    state = step(state=state, action=25)
    state = step(state=state, action=8)
    state = step(state=state, action=25)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=14)  # BLACK
    assert count_ji(state._x, BLACK, BOARD_SIZE) == 17
    assert count_ji(state._x, WHITE, BOARD_SIZE) == 0
    state = step(state=state, action=24)  # WHITE
    assert count_ji(state._x, BLACK, BOARD_SIZE) == 5
    assert count_ji(state._x, WHITE, BOARD_SIZE) == 0

    # =====
    # + @ @ @ +
    # @ O + O @
    # + @ @ @ +
    # + + + + +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=3)
    state = step(state=state, action=25)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    state = step(state=state, action=9)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=25)  # BLACK
    assert count_ji(state._x, BLACK, BOARD_SIZE) == 14
    assert count_ji(state._x, WHITE, BOARD_SIZE) == 0

    # =====
    # + + O + +
    # + O @ O +
    # O @ + @ O
    # + O @ O +
    # + + O + +
    state = init(key=key)
    state = step(state=state, action=7)  # BLACK
    state = step(state=state, action=2)  # WHITE
    state = step(state=state, action=11)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=17)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=16)
    state = step(state=state, action=25)
    state = step(state=state, action=18)
    state = step(state=state, action=25)
    state = step(state=state, action=22)  # WHITE
    assert count_ji(state._x, BLACK, BOARD_SIZE) == 1
    assert count_ji(state._x, WHITE, BOARD_SIZE) == 12

    # =====
    # + @ @ @ +
    # @ O @ + @
    # @ O @ O @
    # @ O @ O @
    # @ O O O @
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=6)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=11)
    state = step(state=state, action=3)
    state = step(state=state, action=13)
    state = step(state=state, action=5)
    state = step(state=state, action=16)
    state = step(state=state, action=7)
    state = step(state=state, action=18)
    state = step(state=state, action=9)
    state = step(state=state, action=21)
    state = step(state=state, action=10)
    state = step(state=state, action=22)
    state = step(state=state, action=12)
    state = step(state=state, action=23)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=24)
    state = step(state=state, action=25)
    state = step(state=state, action=20)
    state = step(state=state, action=25)
    assert count_ji(state._x, BLACK, BOARD_SIZE) == 2
    assert count_ji(state._x, WHITE, BOARD_SIZE) == 0
    state = step(state=state, action=8)
    assert count_ji(state._x, BLACK, BOARD_SIZE) == 10
    assert count_ji(state._x, WHITE, BOARD_SIZE) == 0

    # セキ判定
    # =====
    # + @ O + +
    # O @ O + +
    # O @ O + +
    # O @ O + +
    # + @ O + +
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=2)  # WHITE
    state = step(state=state, action=6)
    state = step(state=state, action=5)
    state = step(state=state, action=11)
    state = step(state=state, action=7)
    state = step(state=state, action=16)
    state = step(state=state, action=10)
    state = step(state=state, action=21)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=22)
    assert count_ji(state._x, BLACK, BOARD_SIZE) == 0
    assert count_ji(state._x, WHITE, BOARD_SIZE) == 10

    # =====
    # O O O O +
    # O @ @ @ O
    # O @ + @ O
    # O @ @ @ O
    # + O O O +
    state = init(key=key)
    state = step(state=state, action=6)  # BLACK
    state = step(state=state, action=0)  # WHITE
    state = step(state=state, action=7)
    state = step(state=state, action=1)
    state = step(state=state, action=8)
    state = step(state=state, action=2)
    state = step(state=state, action=11)
    state = step(state=state, action=3)
    state = step(state=state, action=13)
    state = step(state=state, action=5)
    state = step(state=state, action=16)
    state = step(state=state, action=9)
    state = step(state=state, action=17)
    state = step(state=state, action=10)
    state = step(state=state, action=18)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=21)
    state = step(state=state, action=25)
    state = step(state=state, action=22)
    state = step(state=state, action=25)
    state = step(state=state, action=23)
    assert count_ji(state._x, BLACK, BOARD_SIZE) == 1
    assert count_ji(state._x, WHITE, BOARD_SIZE) == 3

    # =====
    # + + + + +
    # + + + + +
    # + + + + +
    # + + + + +
    # + + + + +
    state = init(key=key)
    #assert count_ji(state, 0, BOARD_SIZE) == 0
    #assert count_ji(state, 1, BOARD_SIZE) == 0

    # =====
    # + + + + +
    # + @ @ @ +
    # + @ + @ +
    # + @ @ @ +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=6)
    state = step(state=state, action=25)
    state = step(state=state, action=7)
    state = step(state=state, action=25)
    state = step(state=state, action=8)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=13)
    state = step(state=state, action=25)
    state = step(state=state, action=16)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=18)
    state = step(state=state, action=25)
    assert count_ji(state._x, BLACK, BOARD_SIZE) == 17
    assert count_ji(state._x, WHITE, BOARD_SIZE) == 0


def test_counting_point():
    key = jax.random.PRNGKey(0)
    count_scores = jax.jit(_count_scores, static_argnums=(1,))
    # =====
    # @ + @ + @
    # + @ + @ +
    # @ + @ + @
    # + + + + +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=0)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=4)
    state = step(state=state, action=25)
    state = step(state=state, action=6)
    state = step(state=state, action=25)
    state = step(state=state, action=8)
    state = step(state=state, action=25)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=14)  # BLACK
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([25, 0], dtype=jnp.float32))
    state = step(state=state, action=24)  # WHITE
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([13, 1], dtype=jnp.float32))

    # =====
    # + @ @ @ +
    # @ O + O @
    # + @ @ @ +
    # + + + + +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=25)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=25)
    state = step(state=state, action=3)
    state = step(state=state, action=25)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    state = step(state=state, action=9)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=12)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=25)  # BLACK
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([22, 2], dtype=jnp.float32))

    # =====
    # + + O + +
    # + O @ O +
    # O @ + @ O
    # + O @ O +
    # + + O + +
    state = init(key=key)
    state = step(state=state, action=7)  # BLACK
    state = step(state=state, action=2)  # WHITE
    state = step(state=state, action=11)
    state = step(state=state, action=6)
    state = step(state=state, action=13)
    state = step(state=state, action=8)
    state = step(state=state, action=17)
    state = step(state=state, action=10)
    state = step(state=state, action=25)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=16)
    state = step(state=state, action=25)
    state = step(state=state, action=18)
    state = step(state=state, action=25)
    state = step(state=state, action=22)  # WHITE
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([5, 20], dtype=jnp.float32))

    # =====
    # + @ @ @ +
    # @ O @ + @
    # @ O @ O @
    # @ O @ O @
    # @ O O O @
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=6)  # WHITE
    state = step(state=state, action=2)
    state = step(state=state, action=11)
    state = step(state=state, action=3)
    state = step(state=state, action=13)
    state = step(state=state, action=5)
    state = step(state=state, action=16)
    state = step(state=state, action=7)
    state = step(state=state, action=18)
    state = step(state=state, action=9)
    state = step(state=state, action=21)
    state = step(state=state, action=10)
    state = step(state=state, action=22)
    state = step(state=state, action=12)
    state = step(state=state, action=23)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=24)
    state = step(state=state, action=25)
    state = step(state=state, action=20)
    state = step(state=state, action=25)
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([16, 8], dtype=jnp.float32))
    state = step(state=state, action=8)
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([25, 0], dtype=jnp.float32))

    # セキ判定
    # =====
    # + @ O + +
    # O @ O + +
    # O @ O + +
    # O @ O + +
    # + @ O + +
    state = init(key=key)
    state = step(state=state, action=1)  # BLACK
    state = step(state=state, action=2)  # WHITE
    state = step(state=state, action=6)
    state = step(state=state, action=5)
    state = step(state=state, action=11)
    state = step(state=state, action=7)
    state = step(state=state, action=16)
    state = step(state=state, action=10)
    state = step(state=state, action=21)
    state = step(state=state, action=12)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=22)
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([5, 18], dtype=jnp.float32))

    # =====
    # O O O O +
    # O @ @ @ O
    # O @ + @ O
    # O @ @ @ O
    # + O O O +
    state = init(key=key)
    state = step(state=state, action=6)  # BLACK
    state = step(state=state, action=0)  # WHITE
    state = step(state=state, action=7)
    state = step(state=state, action=1)
    state = step(state=state, action=8)
    state = step(state=state, action=2)
    state = step(state=state, action=11)
    state = step(state=state, action=3)
    state = step(state=state, action=13)
    state = step(state=state, action=5)
    state = step(state=state, action=16)
    state = step(state=state, action=9)
    state = step(state=state, action=17)
    state = step(state=state, action=10)
    state = step(state=state, action=18)
    state = step(state=state, action=14)
    state = step(state=state, action=25)
    state = step(state=state, action=15)
    state = step(state=state, action=25)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=21)
    state = step(state=state, action=25)
    state = step(state=state, action=22)
    state = step(state=state, action=25)
    state = step(state=state, action=23)
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([9, 16], dtype=jnp.float32))

    # =====
    # + + + + +
    # + + + + +
    # + + + + +
    # + + + + +
    # + + + + +
    state = init(key=key)
    # 本当は[0, 0]
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([25, 25], dtype=jnp.float32))

    # =====
    # + + + + +
    # + @ @ @ +
    # + @ + @ +
    # + @ @ @ +
    # + + + + +
    state = init(key=key)
    state = step(state=state, action=6)
    state = step(state=state, action=25)
    state = step(state=state, action=7)
    state = step(state=state, action=25)
    state = step(state=state, action=8)
    state = step(state=state, action=25)
    state = step(state=state, action=11)
    state = step(state=state, action=25)
    state = step(state=state, action=13)
    state = step(state=state, action=25)
    state = step(state=state, action=16)
    state = step(state=state, action=25)
    state = step(state=state, action=17)
    state = step(state=state, action=25)
    state = step(state=state, action=18)
    state = step(state=state, action=25)
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([25, 0], dtype=jnp.float32))
    # =====
    # + @ @ O +
    # + + @ O +
    # + + @ O O
    # + + @ O O
    # + + @ O O
    # Tromp-Taylor rule: Black 15, White 10 → White Win
    # Japanese rule: Black 9, White 2 → Black Win
    state = init(key=key)
    state = step(state=state, action=1)
    state = step(state=state, action=3)
    state = step(state=state, action=2)
    state = step(state=state, action=8)
    state = step(state=state, action=7)
    state = step(state=state, action=13)
    state = step(state=state, action=12)
    state = step(state=state, action=14)
    state = step(state=state, action=17)
    state = step(state=state, action=18)
    state = step(state=state, action=22)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=23)
    state = step(state=state, action=25)
    state = step(state=state, action=24)
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([15, 10], dtype=jnp.float32))

    # =====
    # + @ @ O +
    # @ + @ O +
    # + + @ O O
    # + + @ O O
    # + + @ O O
    # Agehama: Black 1, White 0
    # Tromp-Taylor rule: Black 15, White 10 → White Win
    # Japanese rule: Black 9, White 2 → Black Win
    state = init(key=key)
    state = step(state=state, action=1)
    state = step(state=state, action=3)
    state = step(state=state, action=2)
    state = step(state=state, action=8)
    state = step(state=state, action=7)
    state = step(state=state, action=13)
    state = step(state=state, action=12)
    state = step(state=state, action=14)
    state = step(state=state, action=17)
    state = step(state=state, action=18)
    state = step(state=state, action=22)
    state = step(state=state, action=19)
    state = step(state=state, action=25)
    state = step(state=state, action=23)
    state = step(state=state, action=25)
    state = step(state=state, action=24)
    state = step(state=state, action=25)
    state = step(state=state, action=0)
    state = step(state=state, action=5)
    state = step(state=state, action=25)
    assert jnp.all(count_scores(state._x, BOARD_SIZE) == jnp.array([15, 10], dtype=jnp.float32))


def test_PSK():
    env = Go(size=5)
    env.init = jax.jit(env.init)
    env.step = jax.jit(env.step)
    state = env.init(jax.random.PRNGKey(0))
    state = env.step(state, 20)  # BLACK
    state = env.step(state, 17)  # WHITE
    state = env.step(state, 6)  # BLACK
    state = env.step(state, 9)  # WHITE
    state = env.step(state, 11)  # BLACK
    state = env.step(state, 1)  # WHITE
    state = env.step(state, 25)  # BLACK
    state = env.step(state, 4)  # WHITE
    state = env.step(state, 24)  # BLACK
    state = env.step(state, 19)  # WHITE
    state = env.step(state, 16)  # BLACK
    state = env.step(state, 18)  # WHITE
    state = env.step(state, 5)  # BLACK
    state = env.step(state, 0)  # WHITE
    state = env.step(state, 3)  # BLACK
    state = env.step(state, 21)  # WHITE
    state = env.step(state, 12)  # BLACK
    state = env.step(state, 13)  # WHITE
    state = env.step(state, 7)  # BLACK
    state = env.step(state, 8)  # WHITE
    state = env.step(state, 22)  # BLACK
    state = env.step(state, 25)  # WHITE
    state = env.step(state, 21)  # BLACK
    state = env.step(state, 23)  # WHITE
    state = env.step(state, 10)  # BLACK
    #  O O + @ O
    #  @ @ @ O O
    #  @ @ @ O +
    #  + @ O O O
    #  @ @ @ O +
    state = env.step(state, 2)  # WHITE
    state = env.step(state, 3)  # BLACK
    state = env.step(state, 0)  # WHITE
    assert not state.terminated
    state = env.step(state, 25)  # BLACK
    assert not state.terminated
    state = env.step(state, 1)  # WHITE
    #  O O + @ O
    #  @ @ @ O O
    #  @ @ @ O +
    #  + @ O O O
    #  @ @ @ O +
    assert state.terminated
    # assert state._x._black_player == 1
    assert (state.rewards == jnp.float32([-1, 1])).all()  # black wins


def test_max_step_termination():
    env = Go(size=9, max_terminal_steps=10)
    init_fn = jax.jit(env.init)
    step_fn = jax.jit(env.step)
    state = init_fn(jax.random.PRNGKey(0))
    for i in range(10):
        assert not state.terminated
        state = step_fn(state, i)
    assert state.terminated
    assert not (state.rewards == jnp.float32([0, 0])).all()  # should not tie


def test_env_id():
    env = Go(size=9)
    init_fn = jax.jit(env.init)
    state = init_fn(jax.random.PRNGKey(0))
    assert state.env_id == "go_9x9"
    init_fn = jax.jit(jax.vmap(env.init))
    state = init_fn(jax.random.split(jax.random.PRNGKey(0)))
    assert state.env_id == "go_9x9"

    env = Go(size=19)
    init_fn = jax.jit(env.init)
    state = init_fn(jax.random.PRNGKey(0))
    assert state.env_id == "go_19x19"
    init_fn = jax.jit(jax.vmap(env.init))
    state = init_fn(jax.random.split(jax.random.PRNGKey(0)))
    assert state.env_id == "go_19x19"

    env = Go(size=5)
    init_fn = jax.jit(env.init)
    state = init_fn(jax.random.PRNGKey(0))
    assert state.env_id == "go_5x5"
    init_fn = jax.jit(jax.vmap(env.init))
    state = init_fn(jax.random.split(jax.random.PRNGKey(0)))
    assert state.env_id == "go_5x5"


def test_api():
    import pgx
    env = pgx.make("go_9x9")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)
    env = pgx.make("go_19x19")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)
