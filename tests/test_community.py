"""Tests for the community detection features and the fate statistic.

The fate statistic is counted in closed form rather than by enumerating pairs,
because the MCU alone has 6.1 million character pairs and the permutation loop
runs the count 2,000 times. Closed-form counting is easy to get subtly wrong, so
the first test checks it against brute-force enumeration on small inputs.
"""

import itertools

import networkx as nx
import numpy as np
import pytest

from src.q1_character.community import CommunityFate
from src.q1_character.features import FeatureBuilder


def brute_force_d(community, dead):
    """The same statistic, counted by walking every pair."""
    same_community_same_fate = same_community = 0
    other_same_fate = other = 0
    for i, j in itertools.combinations(range(len(community)), 2):
        shares_fate = dead[i] == dead[j]
        if community[i] == community[j]:
            same_community += 1
            same_community_same_fate += shares_fate
        else:
            other += 1
            other_same_fate += shares_fate
    if not same_community or not other:
        return float("nan")
    return same_community_same_fate / same_community - other_same_fate / other


@pytest.mark.parametrize("community,dead", [
    ([0, 0, 1, 1], [1, 1, 0, 0]),
    ([0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0]),
    ([0, 1, 2, 0, 1, 2, 0], [1, 1, 0, 0, 1, 0, 1]),
    ([0, 0, 1, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1, 0]),
])
def test_fate_difference_matches_brute_force(community, dead):
    fast = CommunityFate.fate_difference(np.array(community), np.array(dead, float))
    slow = brute_force_d(community, dead)
    assert fast == pytest.approx(slow, abs=1e-12)


def test_perfectly_segregated_communities_give_d_of_one():
    """Two communities, one all dead and one all alive.

    Every within-community pair shares a fate and no cross-community pair does,
    so D is exactly 1. This is the upper bound of the statistic.
    """
    community = np.array([0, 0, 0, 1, 1, 1])
    dead = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert CommunityFate.fate_difference(community, dead) == pytest.approx(1.0)


def test_uniform_fate_gives_zero():
    """If nobody dies, community membership cannot predict fate."""
    community = np.array([0, 0, 1, 1, 2, 2])
    dead = np.zeros(6)
    assert CommunityFate.fate_difference(community, dead) == pytest.approx(0.0)


def test_single_community_is_undefined():
    """With one community there are no cross-community pairs to compare to."""
    community = np.array([0, 0, 0, 0])
    dead = np.array([1.0, 0.0, 1.0, 0.0])
    assert np.isnan(CommunityFate.fate_difference(community, dead))


def test_permutation_null_is_centred_near_zero():
    """Shuffling labels should destroy any relationship with the partition."""
    rng = np.random.default_rng(0)
    community = np.repeat(np.arange(8), 8)
    dead = (rng.random(64) < 0.4).astype(float)

    result = CommunityFate.test_franchise(community, dead, rng)
    assert abs(result["d_null_mean"]) < 0.02
    assert 0.0 < result["p_permutation"] <= 1.0


def _two_clique_graph():
    """Two 4-cliques joined by a single edge: an unambiguous partition."""
    graph = nx.DiGraph()
    left, right = "abcd", "wxyz"
    for group in (left, right):
        for source, target in itertools.permutations(group, 2):
            graph.add_edge(source, target)
    graph.add_edge("d", "w")
    return graph


def test_louvain_recovers_two_obvious_communities():
    columns, summary = FeatureBuilder.communities(_two_clique_graph())

    assert summary["n_communities"] == 2
    # Two near-separate cliques sit well above the 0.3 interpretability floor.
    assert summary["modularity"] > 0.3
    assert columns["a"]["community_id"] == columns["b"]["community_id"]
    assert columns["a"]["community_id"] != columns["w"]["community_id"]


def test_embeddedness_is_a_share():
    columns, _ = FeatureBuilder.communities(_two_clique_graph())
    for values in columns.values():
        assert 0.0 <= values["embeddedness"] <= 1.0
        assert 0.0 <= values["community_rank"] <= 1.0
        assert 0.0 < values["community_share"] <= 1.0

    # "d" is the only node with an edge leaving its clique, so it is the least
    # embedded member of that community.
    clique = [n for n in "abcd"]
    assert columns["d"]["embeddedness"] == min(
        columns[n]["embeddedness"] for n in clique
    )


def test_edgeless_graph_reports_undefined_modularity():
    """Modularity is undefined without edges, and is not reported as zero."""
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b", "c"])
    columns, summary = FeatureBuilder.communities(graph)

    assert summary["n_communities"] == 3
    assert np.isnan(summary["modularity"])
    assert all(v["community_size"] == 1 for v in columns.values())


def test_empty_graph_returns_nothing():
    columns, summary = FeatureBuilder.communities(nx.DiGraph())
    assert columns == {}
    assert summary == {}
