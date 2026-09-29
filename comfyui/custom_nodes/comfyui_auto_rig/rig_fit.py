"""
Fit a Mixamo-named humanoid skeleton to an arbitrary mesh, and skin it.

Deliberately free of any ComfyUI imports so the fit can be exercised standalone
(see scripts/test-autorig.py in the project repo).

Why landmark fitting rather than a learned auto-rigger
-----------------------------------------------------
Learned riggers (UniRig, RigNet, MagicArticulate) predict *a* skeleton, not
*the* Mixamo skeleton: bone count, names and hierarchy vary per mesh, so nothing
downstream can bind a Mixamo clip to them without a hand-built retarget map.
Mixamo animation tracks are addressed strictly by joint name, so producing the
exact `mixamorig:*` hierarchy is the whole game -- and for a roughly A/T-posed
humanoid that is a solvable geometry problem needing no model download.

The fit is pose-independent because it works on the mesh's own connectivity:
geodesic farthest-point sampling from the body centre lands on the five
extremities (head, two hands, two feet) whatever the limb angles, and the
shortest paths back to the root trace the limbs themselves.
"""

import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

# ------------------------------------------------------------------ skeleton --
# The core Mixamo humanoid. Fingers are omitted: Mixamo body clips bind by name
# and simply leave finger tracks unresolved, which is the right trade for a
# low-poly game character.
MIXAMO_JOINTS = [
    ("Hips", -1),
    ("Spine", 0), ("Spine1", 1), ("Spine2", 2),
    ("Neck", 3), ("Head", 4), ("HeadTop_End", 5),
    ("LeftShoulder", 3), ("LeftArm", 7), ("LeftForeArm", 8), ("LeftHand", 9),
    ("RightShoulder", 3), ("RightArm", 11), ("RightForeArm", 12), ("RightHand", 13),
    ("LeftUpLeg", 0), ("LeftLeg", 15), ("LeftFoot", 16),
    ("LeftToeBase", 17), ("LeftToe_End", 18),
    ("RightUpLeg", 0), ("RightLeg", 20), ("RightFoot", 21),
    ("RightToeBase", 22), ("RightToe_End", 23),
]

J = {n: i for i, (n, _) in enumerate(MIXAMO_JOINTS)}
NUM_JOINTS = len(MIXAMO_JOINTS)

# Farthest-point picks used to locate the limbs. Five would be exactly the five
# extremities of a bare humanoid, but a game character has ears, a snout, a tail
# or a held prop that are just as far along the surface, so the search takes a
# surplus and classifies afterwards.
NUM_EXTREMITY_PICKS = 10

# Mirror pairs, used to force the rig bilaterally symmetric. An asymmetric rig
# makes every Mixamo clip visibly lopsided even on a symmetric mesh.
MIRROR_PAIRS = [
    (J["LeftShoulder"], J["RightShoulder"]), (J["LeftArm"], J["RightArm"]),
    (J["LeftForeArm"], J["RightForeArm"]), (J["LeftHand"], J["RightHand"]),
    (J["LeftUpLeg"], J["RightUpLeg"]), (J["LeftLeg"], J["RightLeg"]),
    (J["LeftFoot"], J["RightFoot"]), (J["LeftToeBase"], J["RightToeBase"]),
    (J["LeftToe_End"], J["RightToe_End"]),
]
CENTRE_JOINTS = [J["Hips"], J["Spine"], J["Spine1"], J["Spine2"],
                 J["Neck"], J["Head"], J["HeadTop_End"]]


# ------------------------------------------------------------------- helpers --
def weld(verts, faces, tol_frac=1e-5):
    """Merge coincident vertices, returning (welded verts, welded faces, map).

    Essential, not an optimisation. A UV-unwrapped mesh duplicates every vertex
    that sits on a chart seam -- one copy per chart, same position, different UV
    -- so the finished mesh is topologically shattered into hundreds of islands
    even though it looks solid. Every part of this fitter walks the edge graph,
    so on an unwelded mesh the geodesic search explores one UV chart and reports
    a confident rig for a fragment of a shoulder.

    `map` sends each original vertex to its welded index, so skin weights solved
    on the welded mesh expand straight back onto the original vertex order.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    span = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) or 1.0
    q = np.round(verts / (tol_frac * span)).astype(np.int64)
    _uniq, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    wverts = verts[first]
    wfaces = inv[faces]
    # Faces that collapsed to a line or point carry no connectivity.
    keep = ((wfaces[:, 0] != wfaces[:, 1]) & (wfaces[:, 1] != wfaces[:, 2])
            & (wfaces[:, 0] != wfaces[:, 2]))
    return wverts, wfaces[keep], inv


def edge_graph(verts, faces):
    """Undirected vertex graph with Euclidean edge weights, as CSR."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.concatenate([e, e[:, ::-1]], axis=0)
    w = np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)
    n = len(verts)
    return coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()


def _largest_component(graph, n):
    """Mask of the biggest connected component, plus the component labels.

    Generated meshes routinely carry small floating shells; running the geodesic
    analysis over those would put an 'extremity' on a stray fragment.
    """
    ncomp, labels = connected_components(graph, directed=False)
    if ncomp == 1:
        return np.ones(n, dtype=bool), labels, 1
    sizes = np.bincount(labels)
    return labels == int(np.argmax(sizes)), labels, ncomp


def _symmetry_plane(verts, sample=6000, seed=0, search=0.15, steps=17):
    """Find the plane x = c that best mirrors the mesh onto itself.

    The midpoint of the bounding box is not that plane: one prop held out to the
    side -- a lantern, a satchel, a weapon -- moves it by half the prop's reach,
    and every left/right decision downstream inherits the error. Scoring actual
    mirror residual instead is barely more expensive and is not fooled by
    anything that does not have a matching partner on the other side.

    Returns (centre, mean residual distance).
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return float(0.5 * (verts[:, 0].min() + verts[:, 0].max())), 0.0

    rs = np.random.RandomState(int(seed))
    v = verts if len(verts) <= sample else verts[rs.choice(len(verts), sample, replace=False)]
    tree = cKDTree(v)
    span = float(verts[:, 0].max() - verts[:, 0].min()) or 1.0
    # Centred on the median rather than the box centre: half the vertices lie on
    # each side of the median whatever a single outlying part is doing.
    med = float(np.median(verts[:, 0]))

    best_c, best_s = med, None
    for c in np.linspace(med - search * span, med + search * span, int(steps)):
        m = v.copy()
        m[:, 0] = 2.0 * c - m[:, 0]
        s = float(np.mean(tree.query(m)[0]))
        if best_s is None or s < best_s:
            best_c, best_s = float(c), s
    return best_c, float(best_s or 0.0)


def _chain_to(pred, target):
    """Walk a Dijkstra predecessor array back to the source. Returns root->target."""
    out = []
    v = int(target)
    seen = set()
    while v >= 0 and v not in seen:
        out.append(v)
        seen.add(v)
        v = int(pred[v])
    return out[::-1]


def _sample_chain(verts, chain, frac):
    """Point at `frac` of the arc length along a vertex chain."""
    pts = verts[chain]
    if len(pts) == 1:
        return pts[0].copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-9:
        return pts[0].copy()
    t = float(np.clip(frac, 0.0, 1.0)) * s[-1]
    return np.array([np.interp(t, s, pts[:, k]) for k in range(3)])


def _medial(verts, p, radius):
    """Pull a point onto the local medial axis by averaging nearby surface.

    A joint sampled from a surface path sits *on* the skin; the centroid of the
    surrounding surface patch sits near the limb's axis, which is where a bone
    belongs. Without this, limbs rotate about their own skin and pinch.
    """
    d = np.linalg.norm(verts - p[None, :], axis=1)
    m = d < radius
    if int(m.sum()) < 8:
        return p
    return verts[m].mean(axis=0)


# --------------------------------------------------------- torso landmarks ---
# The limbs come out of the geodesic analysis, but the torso does not: on a
# *surface* mesh the shortest path to the head and the shortest path to a hand
# leave the pelvis on different sides of the trunk and share almost no prefix,
# so a branch point tells you nothing about where the chest is. The trunk is
# instead read off horizontal cross-sections, which is what actually carries the
# signal -- legs split, the neck pinches, the shoulders flare.

def _central_slice(verts, y, band, cx, span):
    """Vertices in a horizontal band, limited to the run of X occupancy that
    contains the symmetry plane. That drops arms and dangling hands without
    needing a fixed distance cutoff that a wide torso would trip over."""
    m = np.abs(verts[:, 1] - y) < band
    if int(m.sum()) < 8:
        return None
    sel = verts[m]
    xs = sel[:, 0] - cx
    nb = 33
    hist, edges = np.histogram(xs, bins=nb, range=(-span, span))
    occ = hist > 0
    c = nb // 2
    if not occ[c]:
        near = np.flatnonzero(occ)
        if len(near) == 0:
            return None
        c = int(near[np.argmin(np.abs(near - c))])
    a = c
    while a > 0 and occ[a - 1]:
        a -= 1
    b = c
    while b < nb - 1 and occ[b + 1]:
        b += 1
    keep = (xs >= edges[a]) & (xs <= edges[b + 1])
    return sel[keep] if int(keep.sum()) >= 8 else sel


def _crotch_height(graph, verts, foot_l, foot_r, lo, height):
    """Apex of the shortest surface path between the two feet.

    That path climbs the inside of one leg, crosses at the crotch and descends
    the other, so its highest point *is* the crotch by construction. Deriving it
    from connectivity rather than from a gap in the X occupancy means it does not
    matter how narrow the gap between the legs is -- a histogram has to resolve
    that gap, and on a stocky character whose thighs nearly touch it fails and
    silently returns a crotch up in the chest.
    """
    fallback = lo[1] + 0.48 * height
    d, pred = dijkstra(graph, indices=int(foot_l), return_predecessors=True)
    if not np.isfinite(d[int(foot_r)]):
        return fallback
    chain = _chain_to(pred, int(foot_r))
    if len(chain) < 3:
        return fallback
    apex = float(verts[chain][:, 1].max())
    # If the legs are fused the path climbs over the hips or higher, so refuse a
    # result that is not where a crotch can be.
    if not (lo[1] + 0.20 * height <= apex <= lo[1] + 0.72 * height):
        return fallback
    return apex


def _neck_height(verts, cx, y_lo, y_hi, height, span):
    """The narrowest trunk cross-section in a height range: the neck.

    Deliberately independent of where the arms attach, so the chest can be
    floored against the neck afterwards rather than the other way round.
    """
    ys = np.linspace(y_lo, y_hi, 56)
    band = 0.02 * height
    best_y, best_w = None, None
    for y in ys:
        sl = _central_slice(verts, y, band, cx, span)
        if sl is None:
            continue
        w = float(sl[:, 0].max() - sl[:, 0].min()) + float(sl[:, 2].max() - sl[:, 2].min())
        if best_w is None or w < best_w:
            best_w, best_y = w, y
    return best_y if best_y is not None else 0.5 * (y_lo + y_hi)


def _median_filter(a, w=3):
    """Small 1-D median filter, to keep one noisy shell from deciding a joint."""
    n = len(a)
    h = w // 2
    return np.array([np.median(a[max(0, i - h):min(n, i + h + 1)]) for i in range(n)])


def _arm_attach(graph, verts, faces, chain, hand, height, nbins=40):
    """Index along a root->hand chain where the arm meets the trunk.

    Measured by geodesic shell *area* from the hand. Sweeping outward over the
    surface, area accumulates at a rate set by the local circumference: roughly
    flat along a limb, then climbing steeply once the sweep reaches the trunk.

    The area weighting is the crux. Counting vertices per shell instead measures
    vertex density, which on a decimated low-poly mesh has nothing to do with
    geometry -- big flat triangles on a torso contribute fewer vertices than the
    dense detail around a hand, so the profile came out noisy enough to put the
    shoulder halfway down the forearm. Summing triangle area is a real measure of
    surface and is indifferent to how the mesh happens to be tessellated.

    Local thickness probes were tried before this and are worse: a ball picks up
    a longer stretch of a thin limb and reports a similar spread either way, a
    perpendicular slab reports garbage on the hand's end cap, and both are fooled
    by an A-pose hand resting beside a thigh.
    """
    n = len(chain)
    if n < 8:
        return 0
    dh = dijkstra(graph, indices=int(hand))
    dmax = float(dh[chain[0]])
    if not np.isfinite(dmax) or dmax <= 0:
        return 0

    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                         tri[:, 2] - tri[:, 0]), axis=1)
    fd = dh[faces].mean(axis=1)
    ok = np.isfinite(fd) & (fd <= dmax)
    if int(ok.sum()) < 16:
        return 0
    counts, edges = np.histogram(fd[ok], bins=int(nbins), range=(0.0, dmax),
                                 weights=area[ok])
    counts = _median_filter(counts, 3)
    if len(counts) < 8:
        return 0

    # Skip the hand's own cap, where shells are still growing to the tube's
    # circumference and would depress the baseline.
    i0 = max(2, len(counts) // 8)
    win = max(3, len(counts) // 6)
    base = float(np.median(counts[i0:i0 + win]))
    if base <= 0:
        return 0

    hit = None
    for i in range(i0 + win, len(counts) - 1):
        if counts[i] > 2.2 * base and counts[i + 1] > 2.2 * base:
            hit = i
            break
    if hit is None:
        return 0                       # no shoulder found: not a real arm

    arm_len = float(edges[hit])
    # Translate that geodesic radius back into a position on the chain.
    dchain = dh[np.asarray(chain)]
    idx = int(np.argmin(np.abs(dchain - arm_len)))
    return int(np.clip(idx, 0, n - 4))


def _chain_index_at_height(verts, chain, y):
    """First index along a descending chain at or below height y."""
    ys = verts[chain][:, 1]
    below = np.flatnonzero(ys <= y)
    return int(below[0]) if len(below) else len(chain) - 1


def _point_between(verts, chain, a, b, t):
    """Chain point whose projection onto the segment a->b is closest to `t`.

    Used for mid-limb joints instead of a fraction of arc length. The chain runs
    over the skin, so its length is inflated by every bulge it wraps around -- a
    knee taken at "half the arc" from hip to ankle landed just under the hip on a
    character wearing a tunic, because most of the path length was spent crossing
    the garment. Projecting onto the limb's own axis measures progress *down the
    limb*, which is what the fraction was meant to express.
    """
    pts = verts[chain]
    ab = b - a
    d = float(ab @ ab)
    if d < 1e-18:
        return _sample_chain(verts, chain, t)
    proj = ((pts - a[None, :]) @ ab) / d
    return pts[int(np.argmin(np.abs(proj - float(t))))].copy()


# ------------------------------------------------------------------ the fit ---
def fit_humanoid(verts, faces, forward_axis=1.0, symmetrize=True):
    """Fit the Mixamo joint set to `verts`/`faces`.

    Assumes the pipeline's own convention: Y up, subject facing +Z. With
    forward = +Z and up = +Y the subject's own right is cross(+Z, +Y) = -X, so
    vertices with x > centre belong to the subject's LEFT. `forward_axis` flips
    that when the mesh faces -Z.

    Returns joint positions [25,3] plus the diagnostics needed to judge whether
    the fit is trustworthy.
    """
    raw_n = len(verts)
    verts, faces, _wmap = weld(verts, faces)
    n = len(verts)
    notes = []
    fwd = 1.0 if forward_axis >= 0 else -1.0
    if n < raw_n:
        # Worth reporting: a large gap means the mesh arrived UV-split, and it is
        # the welded count that every geodesic step below actually works on.
        notes.append("welded %d duplicate vertices (%d -> %d)"
                     % (raw_n - n, raw_n, n))

    # Components first, and every measurement taken from the kept body only.
    # Taking the bounding box over all vertices lets one stray floating shell off
    # to one side drag the centre with it, after which the whole body sits on a
    # single side of the "symmetry plane" and no limb can be told left from right.
    graph = edge_graph(verts, faces)
    keep, labels, ncomp = _largest_component(graph, n)
    if ncomp > 1:
        notes.append("mesh has %d disconnected shells; fitted on the largest" % ncomp)

    body = verts[keep]
    lo, hi = body.min(axis=0), body.max(axis=0)
    height = float(hi[1] - lo[1])
    if height <= 1e-9:
        raise ValueError("mesh has no vertical extent; is it Y-up?")

    cx, sym_residual = _symmetry_plane(body)
    if sym_residual > 0.05 * height:
        notes.append("subject is only roughly symmetric (mirror residual %.1f%% of height)"
                     % (100 * sym_residual / height))

    idx = np.flatnonzero(keep)
    # Seed the geodesic field near the pelvis: mid-height on the symmetry plane
    # is inside the hips for essentially any humanoid stance.
    seed_pt = np.array([cx, lo[1] + 0.52 * height, float(verts[keep, 2].mean())])
    root = int(idx[np.argmin(np.linalg.norm(verts[keep] - seed_pt[None, :], axis=1))])

    # -- extremities by geodesic farthest-point sampling ----------------------
    # Five picks land on head + 2 hands + 2 feet for a humanoid at any pose,
    # because geodesic distance runs along the body rather than through the air:
    # a hand near the hip is still far away *along the surface*.
    d0, pred0 = dijkstra(graph, indices=root, return_predecessors=True)
    reach = np.isfinite(d0)
    mind = np.where(reach, d0, -1.0)
    # More picks than the five limbs, because a stylized character has other
    # things that are genuinely far along the surface: ears, a snout, a tail, a
    # weapon or lantern held away from the body. With exactly five picks one of
    # those steals a slot and a real hand or foot is never found.
    picks = []
    for _ in range(NUM_EXTREMITY_PICKS):
        cand = np.where(keep, mind, -1.0)
        p = int(np.argmax(cand))
        if cand[p] <= 0:
            break
        picks.append(p)
        dp = dijkstra(graph, indices=p)
        mind = np.minimum(mind, np.where(np.isfinite(dp), dp, np.inf))
    if len(picks) < 5:
        raise ValueError("only %d extremities found; mesh is not humanoid enough "
                         "to rig automatically" % len(picks))

    # Classify by what each limb actually *is* rather than by rank: feet are the
    # lowest point on their own side of the body, hands the most outstretched.
    # Ranking by height alone lets a low-hanging prop outrank a foot.
    def split(centre):
        s = (verts[picks][:, 0] - centre) * fwd
        return ([p for p, k in zip(picks, s) if k > 0],
                [p for p, k in zip(picks, s) if k <= 0])

    lefts, rights = split(cx)
    if not lefts or not rights:
        # The fitted plane still left every extremity on one side. Rather than
        # refuse, split the picks at their own median: a humanoid's limbs come in
        # pairs, so the median of the extremities is a serviceable divider even
        # when the mesh's mass is lopsided.
        fallback = float(np.median(verts[picks][:, 0]))
        lefts, rights = split(fallback)
        if lefts and rights:
            notes.append("symmetry plane fell outside the extremities; split at "
                         "their median instead - check the rig preview")
            cx = fallback
    if not lefts or not rights:
        raise ValueError("could not separate left from right: all %d extremities lie "
                         "on one side. The mesh may not be a symmetric humanoid, or "
                         "it may not be Y-up facing %s." % (len(picks), "+Z" if fwd > 0 else "-Z"))

    def lowest(group):
        return min(group, key=lambda v: verts[v][1])

    def widest(group, exclude):
        pool = [v for v in group if v not in exclude] or group
        return max(pool, key=lambda v: abs(verts[v][0] - cx))

    footL, footR = lowest(lefts), lowest(rights)
    handL, handR = widest(lefts, {footL}), widest(rights, {footR})
    remaining = [p for p in picks if p not in (footL, footR, handL, handR)]
    head = max(remaining or picks, key=lambda v: verts[v][1])

    # -- limb chains ----------------------------------------------------------
    chains = {k: _chain_to(pred0, v) for k, v in
              (("head", head), ("handL", handL), ("handR", handR),
               ("footL", footL), ("footR", footR))}

    Jp = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
    span = float(max(hi[0] - cx, cx - lo[0])) * 1.02

    # -- trunk from cross-sections -------------------------------------------
    crotch_y = _crotch_height(graph, verts, footL, footR, lo, height)
    # The mesh's crotch is where the legs stop being separate surfaces, which on a
    # clothed character is the hem of a tunic or coat rather than the pelvis. Left
    # alone that puts the root bone down by the knees and every hip rotation
    # swings the garment instead of the body, so the pelvis is floored into the
    # band a pelvis can actually occupy.
    hips_y = max(crotch_y + 0.05 * height, lo[1] + 0.34 * height)
    top_y = float(hi[1])

    # The crown tapers to a point and is narrower than any neck, so it has to be
    # excluded from the search or the "narrowest section" is always the scalp.
    neck_y = _neck_height(verts, cx, hips_y + 0.45 * (top_y - hips_y),
                          top_y - 0.08 * height, height, span)

    armL_i = _arm_attach(graph, verts, faces, chains["handL"], handL, height)
    armR_i = _arm_attach(graph, verts, faces, chains["handR"], handR, height)
    # The arms meet the body at the armpit, which sits lower the further the arms
    # angle down; taken literally it would drag Spine2 toward the waist on an
    # A-pose. Floor it against the neck so the chest stays in the upper trunk
    # whatever the pose, and cap it so it never collides with the neck.
    armpit_y = 0.5 * (verts[chains["handL"][armL_i]][1] + verts[chains["handR"][armR_i]][1])
    chest_y = float(np.clip(armpit_y,
                            hips_y + 0.50 * (neck_y - hips_y),
                            hips_y + 0.90 * (neck_y - hips_y)))

    def axis_at(y):
        """Centre of the trunk at height y."""
        sl = _central_slice(verts, y, 0.03 * height, cx, span)
        if sl is None:
            return np.array([cx, y, float(verts[:, 2].mean())])
        return np.array([float(sl[:, 0].mean()), y, float(sl[:, 2].mean())])

    Jp[J["Hips"]] = axis_at(hips_y)
    Jp[J["Spine"]] = axis_at(hips_y + 0.33 * (chest_y - hips_y))
    Jp[J["Spine1"]] = axis_at(hips_y + 0.66 * (chest_y - hips_y))
    Jp[J["Spine2"]] = axis_at(chest_y)
    Jp[J["Neck"]] = axis_at(neck_y)
    Jp[J["Head"]] = axis_at(neck_y + 0.35 * (top_y - neck_y))
    crown = axis_at(top_y - 0.04 * height)
    Jp[J["HeadTop_End"]] = np.array([cx, top_y, crown[2]])

    for side, split, chain_key, names in (
        ("L", armL_i, "handL", ("LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand")),
        ("R", armR_i, "handR", ("RightShoulder", "RightArm", "RightForeArm", "RightHand")),
    ):
        arm = chains[chain_key][split:]
        if len(arm) < 3:
            raise ValueError("arm chain for side %s is degenerate" % side)
        shoulder = _sample_chain(verts, arm, 0.06)
        wrist = _sample_chain(verts, arm, 0.92)
        Jp[J[names[1]]] = shoulder
        Jp[J[names[2]]] = _point_between(verts, arm, shoulder, wrist, 0.5)   # elbow
        Jp[J[names[3]]] = wrist
        # The clavicle runs from the base of the neck out to the shoulder.
        Jp[J[names[0]]] = Jp[J["Spine2"]] + 0.45 * (Jp[J[names[1]]] - Jp[J["Spine2"]])

    for chain_key, names in (
        ("footL", ("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToe_End")),
        ("footR", ("RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToe_End")),
    ):
        leg = chains[chain_key]
        # The hip socket sits over the thigh, not on the centre line. Sampling
        # the chain near the root lands in the middle of the pelvis where the two
        # thighs have not separated yet, so read the thigh axis down at crotch
        # height -- where the legs are provably distinct -- and lift it back up.
        ti = _chain_index_at_height(verts, leg, crotch_y - 0.08 * height)
        thigh = _medial(verts, verts[leg[ti]], 0.07 * height)
        Jp[J[names[0]]] = np.array([thigh[0], 0.5 * (hips_y + crotch_y), thigh[2]])
        ankle = _sample_chain(verts, leg, 0.88)
        Jp[J[names[1]]] = _point_between(verts, leg, Jp[J[names[0]]], ankle, 0.5)  # knee
        Jp[J[names[2]]] = ankle
        # The toe runs forward along the facing axis at ground level, measured
        # from the foot's own vertices so it fits the actual shoe/hoof shape.
        near = np.linalg.norm(verts[:, [0, 2]] - ankle[None, [0, 2]], axis=1) < 0.18 * height
        low = verts[:, 1] < ankle[1] + 0.08 * height
        sel = near & low
        toe_z = (float((verts[sel][:, 2] * fwd).max()) * fwd
                 if int(sel.sum()) > 4 else ankle[2] + fwd * 0.06 * height)
        ground = lo[1] + 0.02 * height
        Jp[J[names[3]]] = np.array([ankle[0], ground, ankle[2] + 0.55 * (toe_z - ankle[2])])
        Jp[J[names[4]]] = np.array([ankle[0], ground, toe_z])

    # -- medial refinement ----------------------------------------------------
    # Limb joints are sampled from a path over the skin, so they need pulling
    # onto the limb axis. The trunk joints are already cross-section centroids
    # and re-averaging them would only drag them toward whichever side carries
    # more triangles. Radii scale with the joint's role: one global radius
    # either fails to centre a thigh or drags a wrist back into the forearm.
    # UpLeg is excluded: it is already placed on the thigh axis, and re-averaging
    # it inside a pelvis-sized ball would pull it back to the centre line.
    radius = {}
    for name in ("LeftArm", "RightArm"):
        radius[J[name]] = 0.09
    for name in ("LeftForeArm", "RightForeArm", "LeftLeg", "RightLeg"):
        radius[J[name]] = 0.07
    for name in ("LeftHand", "RightHand", "LeftFoot", "RightFoot"):
        radius[J[name]] = 0.055
    for ji, r in radius.items():
        Jp[ji] = _medial(verts, Jp[ji], r * height)

    # -- symmetry and centring ------------------------------------------------
    if symmetrize:
        for ji in CENTRE_JOINTS:
            Jp[ji][0] = cx
        for l, r in MIRROR_PAIRS:
            mirrored = np.array([2 * cx - Jp[r][0], Jp[r][1], Jp[r][2]])
            avg = 0.5 * (Jp[l] + mirrored)
            Jp[l] = avg
            Jp[r] = np.array([2 * cx - avg[0], avg[1], avg[2]])

    # -- diagnostics ----------------------------------------------------------
    # A bad rig is far more expensive than a refused one, so report the signals
    # that actually separate a humanoid from a blob.
    span = float(np.linalg.norm(Jp[J["LeftHand"]] - Jp[J["RightHand"]]))
    leg_len = float(np.linalg.norm(Jp[J["LeftUpLeg"]] - Jp[J["LeftFoot"]]))
    arm_len = float(np.linalg.norm(Jp[J["LeftArm"]] - Jp[J["LeftForeArm"]])
                    + np.linalg.norm(Jp[J["LeftForeArm"]] - Jp[J["LeftHand"]]))
    head_h = float(Jp[J["HeadTop_End"]][1] - Jp[J["Neck"]][1])

    # Bands widened well past human proportions on purpose: this pipeline exists
    # for stylized characters, where short legs and stubby arms under a big head
    # are the intended design and not a symptom of a bad fit.
    ratio_leg = leg_len / height
    ratio_arm = arm_len / height
    if not (0.22 <= ratio_leg <= 0.60):
        notes.append("leg length is %.0f%% of height (expected 25-55%%)"
                     % (100 * ratio_leg))
    if not (0.18 <= ratio_arm <= 0.55):
        notes.append("arm length is %.0f%% of height (expected 22-45%%)"
                     % (100 * ratio_arm))
    if head_h < 0.06 * height:
        notes.append("head is very short relative to the body")
    if abs(Jp[J["LeftHand"]][1] - Jp[J["RightHand"]][1]) > 0.12 * height:
        notes.append("hands sit at very different heights; the pose may be asymmetric")

    def band(v, a, b):
        return 1.0 if a <= v <= b else max(0.0, 1.0 - (a - v if v < a else v - b) / (b - a))

    confidence = float(min(band(ratio_leg, 0.22, 0.60), band(ratio_arm, 0.18, 0.55),
                           band(span / height, 0.25, 1.30)))

    return {
        "joints": Jp.astype(np.float32),
        "names": [n for n, _ in MIXAMO_JOINTS],
        "parents": np.array([p for _, p in MIXAMO_JOINTS], dtype=np.int32),
        "height": height,
        "centre_x": cx,
        "forward": fwd,
        "landmarks": {"root": root, "head": head, "handL": handL, "handR": handR,
                      "footL": footL, "footR": footR},
        "confidence": confidence,
        "notes": notes,
    }


# ------------------------------------------------------------------ skinning --
def _bone_segments(joints, parents):
    """Per-joint (tail, head) segment used for distance. Leaves get a short stub
    along the direction they came from, so a hand still has a capsule to own."""
    segs = np.zeros((len(joints), 2, 3), dtype=np.float64)
    children = {i: [] for i in range(len(joints))}
    for i, p in enumerate(parents):
        if p >= 0:
            children[int(p)].append(i)
    for i in range(len(joints)):
        a = joints[i]
        kids = children[i]
        if kids:
            b = joints[kids[0]] if len(kids) == 1 else joints[kids].mean(axis=0)
        else:
            p = int(parents[i])
            d = a - joints[p] if p >= 0 else np.array([0.0, 1.0, 0.0])
            nrm = np.linalg.norm(d)
            b = a + (d / nrm * 0.25 * nrm if nrm > 1e-9 else np.array([0.0, 1e-3, 0.0]))
        segs[i, 0] = a
        segs[i, 1] = b
    return segs


def _point_segment_distance(pts, a, b):
    """Distance from each point to segment a-b."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-18:
        return np.linalg.norm(pts - a[None, :], axis=1)
    t = np.clip(((pts - a[None, :]) @ ab) / denom, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pts - proj, axis=1)


def skin_mesh(verts, faces, joints, parents,
              iterations=48, max_influences=4, falloff=4.0):
    """Per-vertex skin weights, by diffusing a nearest-bone assignment over the
    mesh's own edge graph.

    Straight inverse-distance skinning bleeds across air gaps -- a hand resting
    near a thigh drags the thigh with it -- because Euclidean distance does not
    know the two are separate surfaces. Diffusing along edges instead means
    influence can only travel over connected surface, which removes that class of
    artefact for the cost of a few dozen sparse matrix-vector products.

    This is heat-diffusion skinning (Baran & Popovic) solved by Jacobi iteration
    rather than a direct sparse solve: same fixed point, no per-bone
    factorisation of a 25k-square matrix.
    """
    # Solved on the welded mesh for the same reason the fit is: diffusion travels
    # along edges, and a UV-split mesh has no edges across its chart seams, so
    # every chart would be smoothed in isolation and seams would tear when posed.
    # Weights are expanded back to the original vertex order at the end, which is
    # exact -- duplicates share a position, so they share a weight.
    orig_n = len(verts)
    verts, faces, wmap = weld(verts, faces)
    joints = np.asarray(joints, dtype=np.float64)
    n = len(verts)
    nb = len(joints)

    segs = _bone_segments(joints, parents)
    dist = np.empty((n, nb), dtype=np.float64)
    for b in range(nb):
        dist[:, b] = _point_segment_distance(verts, segs[b, 0], segs[b, 1])

    scale = max(float(verts[:, 1].max() - verts[:, 1].min()), 1e-9)
    eps = 1e-4 * scale

    # Seed: soft inverse-distance, so diffusion starts from something already
    # sensible rather than a hard one-hot that takes many more iterations.
    w = 1.0 / np.power(dist + eps, falloff)
    w /= w.sum(axis=1, keepdims=True)
    seed = w.copy()

    graph = edge_graph(verts, faces)
    # Row-normalised adjacency: one application replaces each vertex's weights
    # with the mean of its neighbours.
    deg = np.asarray(graph.getnnz(axis=1), dtype=np.float64)
    deg[deg == 0] = 1.0
    A = graph.copy()
    A.data = np.ones_like(A.data)
    inv_deg = 1.0 / deg

    lam = 0.55        # diffusion rate
    mu = 0.22         # pull back toward the distance seed, so limbs keep identity
    for _ in range(int(iterations)):
        nb_avg = A.dot(w) * inv_deg[:, None]
        w = (1.0 - lam) * w + lam * nb_avg
        w = (1.0 - mu) * w + mu * seed
        np.maximum(w, 0.0, out=w)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)

    # Prune to the influences a game engine will actually upload.
    k = int(max(1, min(max_influences, nb)))
    top = np.argpartition(-w, k - 1, axis=1)[:, :k]
    tw = np.take_along_axis(w, top, axis=1)
    ordr = np.argsort(-tw, axis=1)
    top = np.take_along_axis(top, ordr, axis=1)
    tw = np.take_along_axis(tw, ordr, axis=1)
    tw /= np.maximum(tw.sum(axis=1, keepdims=True), 1e-12)

    if orig_n != n:
        top, tw = top[wmap], tw[wmap]
    return top.astype(np.uint16), tw.astype(np.float32)
