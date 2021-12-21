import numpy as np
import pandas as pd
from meshparty.meshwork import algorithms


def is_apical(row):
    if 1 in row["pt_tag"]:
        return True
    elif 1 in row["group_tags"]:
        return True
    else:
        return False


def is_incomplete(row):
    if 3 in row["pt_tag"]:
        return True
    elif 3 in row["group_tags"]:
        return True
    else:
        return False


def is_error(row):
    if 2 in row["pt_tag"]:
        return True
    elif 2 in row["group_tags"]:
        return True
    else:
        return False


def is_basal(row):
    if 4 in row["pt_tag"]:
        return True
    elif 4 in row["group_tags"]:
        return True
    else:
        return False


def del_y(row):
    return -(row["pt"][1] - row["soma_pt"][1]) / 1000


def del_r(row):
    return np.linalg.norm(row["pt"][0:2:2] - row["soma_pt"][0:2:2]) / 1000


def rho(row):
    return np.linalg.norm(row["pt"] - row["soma_pt"]) / 1000


def theta(row):
    return np.arccos(row["del_y"] / row["rho"])


def tortuosity(row):
    return row["dist_to_root"] / row["rho"]


def bp_to_root_factory(skind, sk):
    bps = sk.branch_points

    def bp_to_root(skind):
        if skind == -1:
            return 0
        else:
            is_bp = np.isin(bps, sk.path_to_root(skind))
            return np.sum(is_bp)

    return bp_to_root(skind)


def low_proximate_dendrite_mask(nrn, prox=30, path_dens_th=0.1, path_len_th=20):
    """
    Remove large branches near the soma that lack inputs
    """
    orig_mask = nrn.mesh_mask
    dtr = nrn.skeleton.distance_to_root / 1000
    nrn.apply_mask(nrn.skeleton_property_to_mesh(dtr) < 30)
    prox_inds = nrn.end_points.to_mesh_index_base
    nrn.reset_mask(to=orig_mask)

    path_dens = []
    path_len = []
    for pind in prox_inds:
        ds_inds = nrn.downstream_of(pind)
        num_in_ds = nrn.anno.post_syn.filter_query(ds_inds.to_mesh_mask).count
        net_len = nrn.path_length(ds_inds) / 1000
        if net_len > 0:
            path_dens.append(num_in_ds / net_len)
        else:
            path_dens.append(0)
        path_len.append(net_len)

    branch_df = pd.DataFrame(
        {"mesh_ind": prox_inds, "path_dens": path_dens, "path_len": path_len}
    )

    bad_branch = branch_df.query(
        "path_dens < @path_dens_th and path_len > @path_len_th"
    )

    if len(bad_branch) > 0:
        bad_ds = []
        for mind in bad_branch["mesh_ind"]:
            bad_ds.append(nrn.downstream_of(mind).to_mesh_mask)
        dendrite_mask = np.vstack(bad_ds).sum(axis=0) == 0
    else:
        dendrite_mask = np.full(nrn.mesh.n_vertices, True)
    return dendrite_mask


def apply_dendrite_mask(nrn, sq_th=0.7):
    """
    Use either axon/dendrite split or the low_proximate_dendrite_mask function above to get a dendrite mask.
    """
    sq = algorithms.axon_split_quality(
        nrn.anno.is_axon.mesh_index.to_mesh_mask,
        nrn.anno.pre_syn.mesh_index,
        nrn.anno.post_syn.mesh_index,
    )

    if sq > sq_th:
        dendrite_mask = ~nrn.anno.is_axon.mesh_index.to_mesh_mask
    else:
        dendrite_mask = low_proximate_dendrite_mask(nrn)
    nrn.apply_mask(dendrite_mask)
    return nrn


def add_axon_annotation(nrn):
    if "is_axon" not in nrn.anno.table_names:
        if len(nrn.anno.pre_syn) > 0 and len(nrn.anno.post_syn) > 0:
            is_axon, split_quality = algorithms.split_axon_by_synapses(
                nrn, nrn.anno.pre_syn.mesh_index, nrn.anno.post_syn.mesh_index
            )
        else:
            split_quality = -1
            is_axon = np.full(nrn.n_vertices, False)

        nrn.anno.add_annotations("is_axon", is_axon, mask=True)
    pass


def generate_apical_features(nrn):
    sk = nrn.skeleton
    base_inds = sk.child_nodes(sk.root)
    base_points = [sk.vertices[bind] for bind in base_inds]
    prim_br_skinds = sk.downstream_nodes(base_inds)

    dfs = []
    for base_ind, bpoint, branch_inds in zip(
        base_inds.to_skel_index_base, base_points, prim_br_skinds
    ):
        df = pd.DataFrame(
            {
                "root_id": nrn.seg_id,
                "base_skind": base_ind,
                "base_x": bpoint[0],
                "base_y": bpoint[1],
                "base_z": bpoint[2],
                "v_skind_base": branch_inds.to_skel_index_base,
                "skind": branch_inds,
                "tip_skind": [
                    branch_inds[np.argmax(sk.distance_to_root[skind])]
                    for skind in sk.downstream_nodes(branch_inds)
                ],
            }
        )
        df["tip_pt"] = sk.vertices[df["tip_skind"]].tolist()
        dfs.append(df)
    point_features_df = pd.concat(dfs).reset_index(drop=True)

    soma_pt = sk.vertices[sk.root]
    point_features_df["soma_pt"] = [
        r for r in np.broadcast_to(soma_pt.reshape(1, 3), (len(point_features_df), 3))
    ]
    point_features_df["pt"] = [
        r for r in sk.vertices[point_features_df["skind"].values]
    ]
    point_features_df["top_pt_y"] = np.min(sk.vertices[:, 1]) / 1000
    point_features_df["bot_pt_y"] = np.max(sk.vertices[:, 1]) / 1000

    point_features_df["del_y"] = point_features_df.apply(del_y, axis=1)
    point_features_df["del_r"] = point_features_df.apply(del_r, axis=1)
    point_features_df["rho"] = point_features_df.apply(rho, axis=1)
    point_features_df["theta"] = point_features_df.apply(theta, axis=1)

    point_features_df["tip_x"] = point_features_df["tip_pt"].apply(
        lambda x: x[0] / 1000
    )
    point_features_df["tip_y"] = point_features_df["tip_pt"].apply(
        lambda x: x[1] / 1000
    )
    point_features_df["tip_z"] = point_features_df["tip_pt"].apply(
        lambda x: x[2] / 1000
    )

    point_features_df["pt_x"] = point_features_df["pt"].apply(lambda x: x[0] / 1000)
    point_features_df["pt_y"] = point_features_df["pt"].apply(lambda x: x[1] / 1000)
    point_features_df["pt_z"] = point_features_df["pt"].apply(lambda x: x[2] / 1000)

    point_features_df["dist_from_tip"] = (
        sk.distance_to_root[point_features_df["tip_skind"]]
        - sk.distance_to_root[point_features_df["skind"]]
    ) / 1000

    point_features_df["soma_x"] = point_features_df["soma_pt"].apply(
        lambda x: x[0] / 1000
    )
    point_features_df["soma_y"] = point_features_df["soma_pt"].apply(
        lambda x: x[1] / 1000
    )
    point_features_df["soma_z"] = point_features_df["soma_pt"].apply(
        lambda x: x[2] / 1000
    )

    point_features_df["dist_to_root"] = (
        sk.distance_to_root[point_features_df["skind"]] / 1000
    )
    point_features_df["dist_to_root_rel"] = (
        point_features_df["dist_to_root"]
        / point_features_df.query("skind > -1")["dist_to_root"].max()
    )
    point_features_df["tort"] = point_features_df.apply(tortuosity, axis=1)

    point_features_df["bp_to_root"] = point_features_df["skind"].apply(
        lambda x: bp_to_root_factory(x, nrn.skeleton)
    )
    return point_features_df