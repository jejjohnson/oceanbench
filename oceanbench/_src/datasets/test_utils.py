import pytest
from oceanbench._src.datasets.utils import (
    check_lists_equal, 
    check_lists_subset, 
    update_dict_keys, 
    get_patches_size
)




# TODO: Add demo dataset
# TODO: test get_xrda_dims gets dataset/dataarray dims


@pytest.mark.parametrize(
        "list1,list2",[
    ([1, 2, 3, 5], [1, 3, 2, 5]),
    ([1, 3, 2, 5], [1, 3, 2, 5]),
    ([5, 3, 2, 1], [1, 3, 2, 5]),
    ])
def test_check_lists_equal_true(list1, list2):
    check_lists_equal(list1, list2)
    check_lists_equal(list2, list1)


@pytest.mark.parametrize(
        "list1,list2",[
    ([1, 2, 3, 5], [1, 3, 2]),
    ([1, 2, 3, 5], [1, 3, 2, 6]),
    ([1, 2, 3, 5], [1, 2, 3, 4]),
    ])
def test_check_lists_equal_false(list1, list2):

    with pytest.raises(AssertionError):
        check_lists_equal(list1, list2)
        check_lists_equal(list2, list1)
        

@pytest.mark.parametrize(
        "list1,list2",[
    ([1, 2, 3, 5], []),
    ([1, 2, 3, 5], [1,]),
    ([1, 2, 3, 5], [1, 2]),
    ([1, 2, 3, 5], [1, 2, 3]),
    ([1, 2, 3, 5], [1, 2, 3, 5]),
    ])
def test_check_lists_subset_true(list1, list2):
    check_lists_subset(list2, list1)


@pytest.mark.parametrize(
        "list1,list2",[
    ([1, 2, 3, 5], []),
    ([1, 2, 3, 5], [1,]),
    ([1, 2, 3, 5], [1, 2]),
    ([1, 2, 3, 5], [1, 2, 3]),
    ])
def test_check_lists_subset_false(list1, list2):
    with pytest.raises(AssertionError):
        check_lists_equal(list1, list2)


@pytest.mark.parametrize(
        "source,new,correct",[
    ({"x": 1, "y": 1}, {"x": 1, "y": 1}, {"x": 1, "y": 1}),
    ({"x": 1, "y": 1}, {"y": 1, "x": 1}, {"x": 1, "y": 1}),
    ({"x": 10, "y": 100}, {"x": 1, "y": 1}, {"x": 1, "y": 1}),
    ({"x": 1, "y": 1}, {"x": 1}, {"x": 1, "y": 1}),
    ({"x": 1, "y": 1}, {"y": 1}, {"x": 1, "y": 1}),
    ({"x": 1, "y": 1}, {"x": 10}, {"x": 10, "y": 1}),
    ({"x": 50, "y": 100}, {"x": 10}, {"x": 10, "y": 1}),
    ])
def test_update_dict_keys_true(source, new, correct):

    new_update = update_dict_keys(source, new)

    assert new_update == correct


@pytest.mark.parametrize(
        "dims,patches,strides,correct",[
    ({"x": 10}, {}, {}, {"x": 10}),
    ({"x": 10}, {"x": 1}, {}, {"x": 10}),
    ({"x": 10}, {}, {"x": 1}, {"x": 10}),
    ({"x": 10}, {"x": 2}, {}, {"x": 9}),
    ({"x": 10}, {"x": 3}, {}, {"x": 8}),
    ({"x": 10}, {"x": 4}, {}, {"x": 7}),
    ({"x": 10}, {}, {"x": 2}, {"x": 5}),
    ({"x": 10}, {}, {"x": 3}, {"x": 4}),
    ({"x": 10}, {}, {"x": 4}, {"x": 3}),
    ({"x": 10}, {"x": 2}, {"x": 2}, {"x": 5}),
    ({"x": 10}, {"x": 3}, {"x": 3}, {"x": 3}),
    ])
def test_get_patches_size(dims, patches, strides, correct):

    dims_size = get_patches_size(dims, patches, strides)

    assert dims_size == correct
