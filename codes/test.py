from sentiment_analysis import check_property_1
from sentiment_analysis import repair_with_majority_rule

def test_check_property_1():
    # generate test case and oracle
    N = 4

    ### true
    assert check_property_1(0, [0] * N, [0] * N, N)

    assert check_property_1(1, [1] * N, [1] * N, N)

    ### false
    assert not check_property_1(1, [0] * N, [1] * N, N)
    assert not check_property_1(0, [0] * N, [1] * N, N)

    assert not check_property_1(1, [0] * N, [1] * (N + 1), N)


def test_repair_with_majority_rule():

    assert repair_with_majority_rule([0,0,0]) == 0
    assert repair_with_majority_rule([1,1,1]) == 1
    assert repair_with_majority_rule([1,1,0]) == 1
    assert repair_with_majority_rule([0,1,0]) == 0

    