import copy

import numpy as np
import pandas as pd
from multiwrapper import multiprocessing_utils as mu
from nglui import parser

import pcg_skel
from axon_id.models import add_class_12_primary_anno

from .apical_features import (
    generate_apical_features,
)


def peel_sparse_segments(
    nrn,
    threshold=0.1,
    synapse_table="post_syn",
    heuristic_method=False,
    m1=None,
    m2=None,
    remaining_axon=False,
    mask_out_ax=True,
):
    """
    Take all segments, and iteratively remove segments that are both tips and have a low
    synapse density.
    """
    segs = copy.deepcopy(nrn.skeleton.segments)
    segs_base = [s.to_skel_index_base for s in segs]
    path_inds = [
        np.append([x[0]], nrn.skeleton.parent_nodes(x)) for x in nrn.skeleton.segments
    ]

    if heuristic_method:
        pl = np.array([(nrn.path_length(path) + 1) / 1000 for path in path_inds])
        num_syn = np.array(
            [nrn.anno[synapse_table].filter_query(s.to_mesh_mask).count for s in segs]
        )
        syn_dens = num_syn / pl

    has_root = np.array([nrn.skeleton.root in s for s in segs])

    removed_segs = 1
    total_removed = 0

    if not heuristic_method:
        nrn, seg_ax_map = add_class_12_primary_anno(nrn, m1, m2, mask_out_ax=False)
        # now make seg map True for dendrite and False for axon
        seg_dend_map = np.array(seg_ax_map)
    removed_array = np.array([False] * len(nrn.skeleton.segments))
    while removed_segs > 0:
        has_tip = np.array(
            [
                np.any(np.isin(sb, nrn.skeleton.end_points.to_skel_index_base))
                for sb in segs_base
            ]
        )
        valid_segs = np.logical_and(has_tip, ~has_root)
        if heuristic_method:
            remove_segments = np.logical_and(syn_dens <= threshold, valid_segs)
        else:
            remove_segments = np.logical_and(seg_dend_map, valid_segs)
            removed_array = removed_array ^ remove_segments
        if remove_segments.sum() > 0:
            mask_array = np.vstack(
                [
                    nrn.SkeletonIndex(
                        nrn.skeleton.filter_unmasked_indices(segs_base[x])
                    ).to_mesh_mask
                    for x in np.flatnonzero(remove_segments)
                ]
            )

            mask = np.sum(mask_array, axis=0) == 0
            nrn.apply_mask(mask)
        removed_segs = sum(remove_segments)
        total_removed += removed_segs

    if not mask_out_ax:
        msk = nrn.mesh_mask
        nrn.reset_mask()
        ax_mesh_inds = nrn.mesh_indices[~msk]
        nrn.add_annotations(
            name="is_axon", data=ax_mesh_inds, overwrite=True, mask=True
        )
        # add the ax_mesh as an anno on nrn

    if remaining_axon:
        remaining_axon = np.where(
            ((removed_array) == (seg_dend_map).astype(bool)) == False
        )
        return np.array(segs, dtype="object")[remaining_axon[0]]

    return total_removed


def lookup_level2(row, client, radius=500, voxel_resolution=[4, 4, 40]):
    ap_pt = row["pt_vx"]
    oid = row["root_id"]
    ap_l2id = pcg_skel.chunk_tools.get_closest_lvl2_chunk(
        ap_pt, oid, client=client, radius=radius, voxel_resolution=voxel_resolution
    )
    return ap_l2id


def root_id_client_version(row, timestamp, client):
    l2id = row["l2id"]
    root_id = client.chunkedgraph.get_roots(l2id, timestamp=timestamp)
    return int(root_id)


def _multithread_l2(args):
    x, client, radius, voxel_resolution = args
    return lookup_level2(x, client, radius=radius, voxel_resolution=voxel_resolution)


def generate_nrn_df(
    training_state_ids,
    client,
    radius=500,
    voxel_resolution=[4, 4, 40],
    n_threads=8,
):
    nrn_dfs = []
    for state_id in training_state_ids:
        state = client.state.get_state_json(state_id)

        pts, root_ids = parser.point_annotations(
            state, "apicals", linked_segmentations=True
        )

        new_rids = []
        for rids in root_ids:
            new_rids.append(np.unique(rids))

        nrn_df = pd.DataFrame({"root_id": np.array(new_rids).squeeze(), "pt": pts})
        nrn_df["pt_vx"] = nrn_df["pt"].apply(lambda x: np.array(x).astype(int))
        nrn_df["state_id"] = state_id
        nrn_dfs.append(nrn_df)
    nrn_df = pd.concat(nrn_dfs)

    args = []
    for _, row in nrn_df.iterrows():
        args.append((row, client, radius, voxel_resolution))
    l2s = mu.multithread_func(_multithread_l2, args, n_threads=n_threads)
    # l2s = nrn_df.apply(
    #     lambda x: lookup_level2(
    #         x, client, radius=radius, voxel_resolution=voxel_resolution
    #     ),
    #     axis=1,
    # )
    nrn_df["l2id"] = l2s

    nrn_df["root_id_curr"] = nrn_df.apply(
        lambda x: root_id_client_version(x, client.materialize.get_timestamp(), client),
        axis=1,
    )
    return nrn_df


def process_apical_features(
    nrn,
    preprocessed=False,
    peel=False,
    peel_threshold=None,
    heuristic_method=False,
    peel_table="post_syn",
    m1=None,
    m2=None,
):
    if not preprocessed:
        if peel:
            peel_sparse_segments(
                nrn,
                threshold=peel_threshold,
                synapse_table=peel_table,
                heuristic_method=heuristic_method,
                m1=m1,
                m2=m2,
            )
    point_features_df = generate_apical_features(nrn)
    return point_features_df


def peel_axon_id_apical(nrn, m1, m2, mask_out_ax=True):
    """
    peels axons from neuron with the RF classifiers
    """
    # peel axons from neuron
    remaining_axon = peel_sparse_segments(
        nrn,
        m1=m1,
        m2=m2,
        heuristic_method=False,
        remaining_axon=True,
        mask_out_ax=mask_out_ax,
    )

    # generate apical features
    point_features_df = generate_apical_features(nrn)
    return point_features_df, remaining_axon


def BranchClassifierFactory(rfc, feature_columns):
    class BranchClassifier(object):
        def __init__(
            self,
            min_length,
            logodds_clip,
            softmax_scaler,
            bin_num,
            logodds_thresh,
            softmax_thresh,
            start_distance=0,
        ):
            self.feature_columns = feature_columns
            self.rfc = rfc
            self.min_length = min_length
            self.logodds_clip = logodds_clip
            self.softmax_scaler = softmax_scaler
            self.bin_num = bin_num
            self.logodds_thresh = logodds_thresh
            self.softmax_thresh = softmax_thresh
            self.start_distance = start_distance
            self._data = None

        @property
        def bins(self):
            return np.linspace(0, 1.0, self.bin_num + 1)

        @property
        def bin_linprob_rel(self):
            return np.log10(
                (self.bins[0:-1] + np.finfo(float).eps) / (1 - self.bins[0:-1])
            )

        def log_odds_ratio(self, x):
            bin_inds = np.digitize(x, bins=self.bins, right=False) - 1
            bin_inds[bin_inds < 0] = 0
            bin_inds[bin_inds == len(self.bins) - 1] = len(self.bin_linprob_rel) - 1
            return np.sum(self.bin_linprob_rel[bin_inds])

        def logodds_clipped(self, x):
            return np.clip(x, -self.logodds_clip, self.logodds_clip)

        def softmax_denominator(self, x):
            return np.sum(np.exp(x / self.softmax_scaler))

        def softmax_numerator(self, x):
            return np.exp(x / self.softmax_scaler)

        def apical_softmax(self, x):
            return self.softmax_numerator(x) / self.softmax_denominator(x)

        def fit(self, point_df, base_skind_column, apical_prob_column="apical_prob"):
            self.preprocess_data(
                point_df, base_skind_column, apical_prob_column=apical_prob_column
            )
            self._fit(apical_prob_column)

        def _fit(self, prob_column):
            self._data["logodds_ratio"] = self._data[prob_column].apply(
                self.log_odds_ratio
            )
            self._data["len_br"] = self._data[prob_column].apply(len)
            self._data["logodds_clipped"] = self._data["logodds_ratio"].apply(
                self.logodds_clipped
            )
            self._data["softmax_denom"] = self.softmax_denominator(
                self._data.query(f"len_br > {self.min_length}")["logodds_clipped"]
            )
            self._data["softmax_num"] = self._data["logodds_clipped"].apply(
                self.softmax_numerator
            )
            self._data["apical_softmax"] = (
                self._data["softmax_num"] / self._data["softmax_denom"]
            )

        @property
        def data(self):
            return self._data

        def predict(self):
            if self.data is None:
                return None

            positive_logodds = self.data["logodds_clipped"] > self.logodds_thresh
            softmax_above_thresh = self.data["apical_softmax"] > self.softmax_thresh
            long_enough = self.data["len_br"] > self.min_length
            return np.logical_and(
                positive_logodds, np.logical_and(softmax_above_thresh, long_enough)
            )

        def fit_predict(self, df, base_skind_column, prob_column="apical_prob"):
            self.fit(df, base_skind_column, prob_column)
            self._data["is_apical"] = self.predict()
            return self.data["is_apical"], self.data[base_skind_column]

        def fit_predict_data(self, df, base_skind_column, prob_column="apical_prob"):
            self.fit(df, base_skind_column, prob_column)
            try:
                self._data["is_apical"] = self.predict()
            except:
                self._data["is_apical"] = []
            return self.data

        def preprocess_data(
            self, point_df, base_skind_column, apical_prob_column="apical_prob"
        ):
            df = point_df.copy()
            df = df.query(f"dist_to_root >= {self.start_distance}")
            apical_predict = self.rfc.predict_proba(
                point_df[self.feature_columns].values
            )
            df[apical_prob_column] = apical_predict[:, 1]
            df = (
                df[[base_skind_column, apical_prob_column]]
                .groupby([base_skind_column])
                .agg(list)
                .reset_index()
            )
            self._data = df
            pass

    return BranchClassifier
