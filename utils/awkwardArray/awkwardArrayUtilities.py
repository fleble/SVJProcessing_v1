import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numba as nb
import numpy as np
import awkward as ak
from coffea.processor.accumulator import dict_accumulator, column_accumulator


def to_ak_array(obj):
    """Convert any input type to an awkward array.

    Args:
        obj (any): any type convertible to ak array

    Returns:
        awkward.highlevel.Array
    """

    if isinstance(obj, ak.Array):
        ak_array = obj
    elif isinstance(obj, column_accumulator):
        ak_array = from_column_accumulator(obj)
    elif isinstance(obj, dict_accumulator):
        ak_array = from_dict_accumulator(obj)
    elif isinstance(obj, np.ndarray):
        ak_array = ak.Array(obj)
    elif isinstance(obj, list):
        ak_array = ak.Array(obj)
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64) or isinstance(obj, np.int32) or isinstance(obj, np.int64) \
        or isinstance(obj, float) or isinstance(obj, int):
        ak_array = ak.Array([obj])
    elif isinstance(obj, dict):
        ak_array = from_dict(obj)
    else:
        print("Unknown type %s" %(type(obj)))
        exit(1)

    return ak_array


def as_regular(ak_array):
    """Change type of the axes from var to int.

    Args:
        ak_array (awkward.Array): Non-jagged ak array.

    Returns:
        awkward.Array
    """

    n_axes = get_number_of_axes(ak_array)
    for i_axis in range(1, n_axes):
        ak_array = ak.to_regular(ak_array, axis=i_axis)

    return ak_array


def as_irregular(ak_array):
    """Change type of the axes from int to var.

    Args:
        ak_array (awkward.Array): Non-jagged ak array.

    Returns:
        awkward.Array
    """

    n_axes = get_number_of_axes(ak_array)
    for i_axis in range(1, n_axes):
        ak_array = ak.from_regular(ak_array, axis=i_axis)

    return ak_array


def to_regular(ak_array, fixed_size_dimension=False):
    """Make a regular ak_array by filling it with None.

    Args:
        ak_array (awkward.highlevel.Array): Jagged ak array

    Returns:
        awkward.highlevel.Array
    """

    n_axes = get_number_of_axes(ak_array)
    for i_axis in range(1, n_axes):
        max_length = ak.max(ak.num(ak_array, axis=i_axis), axis=None)
        ak_array = ak.pad_none(ak_array, target=max_length, axis=i_axis, clip=fixed_size_dimension)

    return ak_array


def as_type(ak_array, dtype):
    if ak.count(ak_array) == 0:
        dummy_ak_array = ak.values_astype(ak.Array([[0.]]), dtype)
        ak_array = ak.concatenate((ak_array, dummy_ak_array), axis=0)
        ak_array = ak_array[:-1]
    else:
        ak_array = ak.values_astype(ak.from_iter(ak_array), dtype)

    return ak_array


def flatten(ak_array, axis=None):
    """Flatten ak_array while keeping record fields.

    Args:
        ak_array (awkward.highlevel.Array)
        axis (int or None):
            None: completely flatten the ak array
            int: flatten only the axis with number axis

    Returns:
        awkward.highlevel.Array
    """

    new_ak_array = {}
    for field in ak_array.fields:
        n_axes = get_number_of_axes(ak_array[field])
        new_ak_array[field] = ak.flatten(ak_array[field], axis=axis)
    new_ak_array = ak.Array(new_ak_array)

    return new_ak_array


def unflatten(array, n_nested_levels=1, counts=[2]):
    """Unflatten a regular array.

    Args:
        array (numpy.array or awkward.Array): 1D array
        n_nested_levels (int): Number of nested levels (at least 1!)
        counts (list[int]): The size of the array axis
    
    Returns:
        awkward.Array

    Example:
    >>> ak_array = ak.Array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    >>> flat_array = ak.flatten(ak_array, axis=None)
    >>> flat_array
    <Array [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] type='12 * int64'>
    >>> unflatten(flat_array, n_nested_levels=2, counts=[2, 3])
    <Array [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]] type='2 * var * var * int64'>
    """

    n_total = len(array)
    counts = np.array(counts)
    ak_array = array
    for axis in range(n_nested_levels):
        count_this_axis = np.prod(counts[axis:])
        count = np.ones(int(n_total/count_this_axis), dtype=int) * count_this_axis
        ak_array = ak.unflatten(ak_array, count, axis=axis)

    return ak_array


def from_column_accumulator(accumulator):
    """Return ak array from a coffea column accumulator.

    Args:
        accumulator (coffea.processor.accumulator.column_accumulator)

    Return:
        awkward.Array
    """

    return ak.Array(accumulator.value)


def from_dict_accumulator(accumulator):
    """Return ak array from a coffea dict accumulator.

    Args:
        accumulator (coffea.processor.accumulator.dict_accumulator)

    Return:
        awkward.Array
    """

    array = {}
    for key, value in accumulator.items():
        array[key] = value.value

    return from_dict(array)


def from_dict(dict_):
    """Return ak array from a dict.

    Args:
        dict_ (dict)

    Return:
        awkward.Array
    """

    branch_to_skip = []
    ref = None

    for key, value in dict_.items():
        try:
            if ref is None:
                ref = len(value)
            elif ref != len(value):
                branch_to_skip.append(key)
        except TypeError:
            print(f"Couldn't verify length of branch {key}...")

    if len(branch_to_skip) != 0:
        print(f"Will skip the following branches due to the mismatch in length: {branch_to_skip}")

    array = {}
    for key, value in dict_.items():
        if key in branch_to_skip: continue
        array[key] = to_ak_array(value)

    return ak.Array(array)


def has_nan(ak_array):
    return ak.any(np.isnan(ak_array))


def get_number_of_axes(ak_array):
    """Returns True indice in jagged array.

    Args:
        ak_array (awkward.highlevel.Array):

    Returns:
        int
    """

    return str(ak.type(ak_array)).count("*")


def make_num_filter(num_array, max_num=None):
    """Make filter to make irregular ak array from regular ak array.

    Args:
        num_array (awkward.highlevel.Array): 1 axis with number of objects

    Returns:
        awkward.highlevel.Array

    Examples:
    >>> num_array = ak.Array([2, 3, 1])
    >>> make_num_filter(num_array)
    [[True, True, False], [True, True, True], [True, False, False]]
    """

    if max_num is None:
        max_num = ak.max(num_array, axis=0)
    for idx in range(max_num):
        if idx == 0:
            filter_ = ak.to_numpy(num_array > idx)
        else:
            filter_ = np.vstack((filter_, ak.to_numpy(num_array > idx)))
    
    filter_ = filter_.T
    filter_ = ak.from_iter(filter_)

    if max_num == 1:
        filter_ = ak.singletons(ak.mask(filter_, filter_>-1))

    return filter_


@nb.jit(nopython=True)
def __crop_array_filter_builder(builder, ak_array, target):
    for row in ak_array:
        builder.begin_list()
        for idx, _ in enumerate(row):
            if idx < target:
                builder.append(True)
            else:
                builder.append(False)
        builder.end_list()
    return builder


def crop_array_filter(ak_array, target):
    """Return filter to crop axis-1 of an ak array.

    Args:
        ak_array (ak.Array): 2D ak array
        target (int)
    """

    builder = ak.ArrayBuilder()
    return __crop_array_filter_builder(builder, ak_array, target).snapshot()


def crop_array(ak_array, target):
    """Crop axis-1 of an ak array.

    Args:
        ak_array (ak.Array): 2D ak array
        target (int)
    """

    filter_ = crop_array_filter(ak_array, target)
    return ak_array[filter_]


def get_true_indices(ak_array):
    """Returns True indice in jagged array.

    Args:
        ak_array (awkward.highlevel.Array[bool]): 2D ak array with boolean values
           with structure of a branch of an Events tree or an iterable with
           similar jagged structure.

    Returns:
        list[list[int]]: a jagged list with indices of the jagged array that are true.

    Examples:
        >>> example_ak_array = [ [True, False, True], [True], [], [False, True] ]
        >>> get_true_indices(example_ak_array)
        [ [0, 2], [0], [], [1] ]
    """

    true_indices = [[idx for idx, x in enumerate(y) if x] if y is not None else None for y in ak_array]
    return true_indices

        
def divide_ak_arrays(ak_array1, ak_array2, division_by_zero_value=1., verbose=False):
    """Makes the division of an ak array by another one.
    
    The arrays ak_array1 and ak_array2 must have the same jagged structure.
    If division by zero for some indices, a default value to use can be
    defined, see examples.
    The output array has the same jagged structure as the input ak arrays.
    

    Args:
        ak_array1 (awkward.Array[float])
        ak_array2 (awkward.Array[float]) 
        division_by_zero_value (float, optional, default=1.)
        verbose (bool, optional, default=False)

    Returns:
        awkward.Array[float]: ak_array1 / ak_array2

    Examples:
        >>> ak_array1 = ak.Array([ [0, 3], [5], [1] ])
        >>> ak_array2 = ak.Array([ [3, 3], [0], [2] ])
        >>> divide_ak_arrays(ak_array1, ak_array2)
        [ [0, 1], [1], [0.5] ]
    """

    is_not_zero = (ak_array2!=0.)
    if (not ak.all(is_not_zero)) and verbose:
        print("The following warning about true_divide can be safely ignored.")

    raw_division = ak_array1/ak_array2
    division = ak.where(is_not_zero, raw_division, division_by_zero_value*ak.ones_like(ak_array1))

    # This implementation seems slower:
    #division = ak.Array([ [ x1/x2 if x2 != 0. else division_by_zero_value for x1, x2 in zip(y1, y2) ] for y1, y2 in zip(ak_array1, ak_a0rray2) ])

    return division


def replace(ak_array, value_to_replace, value):
    """Replace value in an ak array by another.

    Args:
        ak_array (ak.Array): ak array without fields
        value_to_replace (float)
        value (float)

    Returns:
        ak.Array
    """

    return ak.fill_none(ak.mask(ak_array, ak_array != value_to_replace), value)


def is_in(array1, array2):
    """
    Args:
        array1 (awkward.highlevel.Array[int])
        array2 (awkward.highlevel.Array[int])

    Returns:
        awkward.highlevel.Array[bool]

    Examples:
        >>> example_array1 = [[0, 1, 1, 2, 2, 3], [0, 0, 1], [0, 1, 1, 2]]
        >>> example_array2 = [[0, 3], [], [0, 2]]
        >>> is_in(example_array1, example_array2)
        [[True, False, False, False, False, True], [False, False, False] [True, False, False, True]]
    """

    return ak.Array([[True if x in y2 else False for x in y1] for y1, y2 in zip(array1, array2)])


def is_in_list(ak_array, list_):
    """Check whether the elements of an ak array are in a list of elements.
    
    Args:
        ak_array1 (awkward.Array[T])
        list_ (list[T])

    Returns:
        awkward.Array[bool]

    Examples:
        >>> ak_array = ak.Array([ [11, 22, -11], [22], [111, 211, -11, 11] ])
        >>> list_ = [11, -11]
        >>> is_in(ak_array, list_)
        [[True, False, True], [False], [False, False, True, True]]
    """

    ak_bool = False * ak.ones_like(ak_array, dtype=bool)
    for el in list_:
        ak_bool = ak_bool | (ak_array == el)

    return ak_bool


def broadcast(ak_array1, ak_array2):
    """Broadcast arrays restricted to their common fields.

    Args:
        ak_array1 (awkward.Array)
        ak_array2 (awkward.Array)

    Returns:
        awkward.Array

    Example:
        >>> ak_array1 = ak.Array({"pt": [400, 300], "phi": [0.2, -0.5]})
        >>> ak_array2 = ak.Array({"pt": [[100, 30], [50, 20, 10]], "eta": [[-2, -1.9], [1.2, 1, 1.5]], "charge": [[1, -1], [1, 0, 0]]})
        >>> ak.to_list(broadcast(ak_array1, ak_array2))
        [[{'pt': [400, 400]}, {'pt': [300, 300, 300]}], [{'pt': [100, 30]}, {'pt': [50, 20, 10]}]]
    """

    ak_array1 = as_irregular(ak_array1)
    ak_array2 = as_irregular(ak_array2)

    name1 = get_ak_array_name(ak_array1)
    name2 = get_ak_array_name(ak_array2)

    fields1 = ak_array1.fields
    fields2 = ak_array2.fields
    if fields1 != fields2:
        common_fields = [ field for field in fields1 if field in fields2 ]
        ak_array1 = ak_array1[common_fields]
        ak_array2 = ak_array2[common_fields]

    broadcasted_arrays = ak.broadcast_arrays(ak_array1, ak_array2)
    if name1 is not None:
        broadcasted_arrays[0] = ak.with_name(broadcasted_arrays[0], name1)
    if name2 is not None:
        broadcasted_arrays[1] = ak.with_name(broadcasted_arrays[1], name2)

    return broadcasted_arrays


def move_axis(ak_array, axis1, axis2):
    """Exchange the position of two axes in an irregular ak array.

    Args:
        ak_array (awkward.highlevel.Array)

    Return:
        awkward.highlevel.Array
    """

    # Regularize array by adding Nones
    ak_array = to_regular(ak_array)

    # Move the axis
    np_array = ak.to_numpy(ak_array)
    np_array = np.moveaxis(np_array, axis1, axis2)
    ak_array = ak.from_iter(np_array)

    # Remove Nones and produce a jagged array
    filter_ = ak.fill_none((ak_array != None), False, axis=-1)
    ak_array = (ak_array[filter_])

    # Remove empty lists
    filter_ = (ak.num(ak_array, axis=2) > 0)
    ak_array = (ak_array[filter_])

    return ak_array


def swap_axes(ak_array):
    """Swap the first two axes of a rectangular ak_array with two axes.

    It is possible that the whole array is copied into the memory, maing this
    function inefficient. No other good laternative was found.

    Args:
        ak_array (ak.Array)

    Returns:
        ak.Array
    """

    np_array = ak.to_numpy(ak_array)
    np_array = np_array.T
    ak_array = ak.Array([np_array])     # Can't avoid that copy in memory!!!
    return ak_array


def ak_to_ptyphim_four_vectors(ak_array):
    """Make a numpy array of pt, rapidity, phi, mass from ak array.

    Args:
        ak_array (ak.Array): 2D ak array with fields pt, rapidity, phi, mass
            and jetIdx. Axis 0 is the event axis, axis 1 is the jet axis.

    Returns:
        np.ndarray: 3D numpy array, where axis 0 is the event axis, axis 1 is
            the constituent axis, and axis 2 is the features (pt, y, phi, m)
            axis. The array represents the 4-vector of all constituents for
            all events for a given jet index.
    """

    list_comprehension = [
       [
           [c.pt, c.rapidity, c.phi, c.mass]
           for c in ak_array[ievent]
       ]
       if ak.count(ak_array[ievent].pt, axis=None) > 0
       else [[1,1,1,1]]
       for ievent in range(ak.num(ak_array, axis=0))
    ]
    np_array = np.array(list_comprehension, dtype=object)

    return np_array


def get_ak_array_name(ak_array):

    name = None
    num = ak.num(ak_array, axis=0)
    # Sometimes, the previous line returns an ak array instead of an int
    if isinstance(num, ak.highlevel.Record):
        num = getattr(num, num.fields[0])
    for not_none_idx in range(num):
        if ak_array[not_none_idx] is not None: break
    if hasattr(ak_array[not_none_idx], "layout"):
        if "__record__" in ak_array[not_none_idx].layout.parameters.keys():
            name = ak_array[not_none_idx].layout.parameters["__record__"]

    return name


def sort_array_with_fields(ak_array, field_name, ascending=False):
    """Sort ak array of records using the values in one of its fields.

    Args:
        ak_array (ak.Array): Ak array of records
        field_name (str): the field to use to sort the array
        ascending (bool, optional, default=False): Set to False to sort objects
            by pT like usually done in HEP (highest pT first)
    """

    sorted_indices = ak.argsort(ak_array[field_name], ascending=ascending)
    return ak_array[sorted_indices]

