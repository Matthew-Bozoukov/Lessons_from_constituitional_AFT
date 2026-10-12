# ABOUTME: Protect honest incomplete judge coverage from silent score imputation or missing cells.
# ABOUTME: Offline checks exercise the explicit exclusion gate without any paid model calls.
import pytest
from scratch.gptoss_swe.odcv_full_closeout import axis_coverage


def test_explicit_exclusion_is_not_a_verdict():
    assert axis_coverage({'a': {'score': 0}, 'b': {'score': 'N/A'}}, {'a','b'}, {'b'}) == 1


@pytest.mark.parametrize('scores,allowed', [
    ({'a': {'score': 1}}, set()),
    ({'a': {'score': 1}, 'b': {'score': 'N/A'}}, set()),
    ({'a': {'score': 1}, 'b': {'score': 1}}, {'b'}),
    ({'a': {'score': 1}, 'b': {'score': float('nan')}}, set()),
    ({'a': {'score': 1}, 'b': {'score': 8}}, set()),
])
def test_incomplete_or_imputed_coverage_refused(scores, allowed):
    with pytest.raises(AssertionError):
        axis_coverage(scores, {'a','b'}, allowed)
