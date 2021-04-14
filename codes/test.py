from sentiment_analysis import check_property_1

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



    