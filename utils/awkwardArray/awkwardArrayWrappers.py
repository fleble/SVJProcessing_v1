import numpy as np
import awkward as ak

import utils.builtinUtilities as builtinUtl
import utils.awkwardArray.awkwardArrayUtilities as akUtl


class AkArrayAsDict(ak.Array):
    """Wrapper of awkward.highlevel.Array defining some dict usual methods.

    Defines:
        keys()
        values()
        items()
        pop(attr)

    Args:
        ak_array (awkward.highlevel.Array)
    """

    def __init__(self, ak_array):
        if not isinstance(ak_array, ak.Array):
            ak_array = ak.Array(ak_array)
        ak.behavior["AkArrayAsDict"] = AkArrayAsDict
        if "__array__" not in ak.parameters(ak_array):
            ak_array = ak.with_parameter(
                ak_array,
                "__array__",
                "AkArrayAsDict",
            )
        super().__init__(ak_array)

    def keys(self):
        return self.fields

    # TODO: There should be a better way using __iter__ and __next__
    def values(self, keys_order=None):
        keys = keys_order or self.keys()
        return (self[attr] for attr in keys)

    def values_as_scalar(self, keys_order=None):
        keys = keys_order or self.keys()
        return (self[attr].to_ak()[0] for attr in keys)

    def items(self):
        return zip(self.keys(), self.values())

    def pop(self, attr):
        array = self[attr]
        self.__init__(self[[key for key in self.keys() if key != attr]])
        return array

    def to_ak(self):
        return ak.with_parameter(self, "__array__", None)
    
    def __hash__(self):
        return hash(self.__repr__())


class AkArrayEnhanced(AkArrayAsDict):

    def __init__(self, ak_array):
        if not isinstance(ak_array, ak.Array):
            ak_array = ak.Array(ak_array)
        ak.behavior["AkArrayEnhanced"] = AkArrayEnhanced
        if "__array__" not in ak.parameters(ak_array):
            ak_array = ak.with_parameter(
                ak_array,
                "__array__",
                "AkArrayEnhanced",
            )
        super().__init__(ak_array)

    def add(self, ak_array):
        """Add new events to the array.

        Args:
            ak_array (awkward.Array): Must have the same fields as self.
        """

        self.__init__(ak.concatenate((self, ak_array), axis=0))

    def add_field(self, field_name, ak_array):
        """Add new field to the array.

        Args:
            field_name (str): name of the new field
            ak_array (awkward.Array): array to add
        """

        self.__init__(ak.with_field(self, ak_array, where=field_name))

    def rename_field(self, field_name, new_field_name):
        """Add new field to the array.

        Args:
            field_name (str): current name of the field
            new_field_name (str): name of the new field
        """

        field = self.pop(field_name)
        self.add_field(new_field_name, field)

    def shuffle(self, indices):
        """Shuffle the array based on input indices.

        Args:
            indices (np.ndarray[int]): new indices
        """

        self.__init__(self[indices])

    def filter(self, filter_array, branch_names_to_filter=".*", counter_branch_name=None):
        """Filter the array.
    
        Args:
            filter_array (ak.Array[bool]): Filter
            branch_names_to_filter (str or list[str]):
                Regex matching branch names (str) or list of branches (list[str])
                with same number of dimensions as the filter, to filter.
            counter_branch_name (str): Branch name that serves as a counter of
                the collection (if filtering a collection).
        """

        # Need to create a new array if not array are sort of changed in place and it breaks
        new_array = {}

        # Need to do this for the filtering of jagged arrays to work
        filter_array = akUtl.as_irregular(filter_array)

        # Making list of branches to filter if provided a regex
        if isinstance(branch_names_to_filter, str):
            all_branch_names = np.array(self.keys())
            indices = builtinUtl.in_regex_list(all_branch_names, branch_names_to_filter)
            branch_names_to_filter = all_branch_names[indices]
    
        # Filtering branches
        for branch_name in branch_names_to_filter:
            new_array[branch_name] = self[branch_name][filter_array]
        
        # Updating counter branch if any
        if counter_branch_name is not None:
            # self[counter_branch_name] = ak.num(self[branch_names[0]], axis=1)
            for branch_name in branch_names_to_filter:
                branch = None
                if akUtl.get_number_of_axes(self[branch_name]) > 1:
                    branch = self[branch_name]
                    break
                if branch is not None:
                    new_array[counter_branch_name] = ak.num(branch, axis=1)

        # Adding branches that were not filtered
        for branch_name in self.keys():
            if branch_name not in branch_names_to_filter and branch_name != counter_branch_name:
                new_array[branch_name] = self[branch_name]

        # Re initializing the object
        self.__init__(new_array)


class AkArrayEnhancedWithOperators(AkArrayEnhanced):

    def __init__(self, ak_array):
        if not isinstance(ak_array, ak.Array):
            ak_array = ak.Array(ak_array)
        ak.behavior["AkArrayEnhancedWithOperators"] = AkArrayEnhancedWithOperators
        if "__array__" not in ak.parameters(ak_array):
            ak_array = ak.with_parameter(
                ak_array,
                "__array__",
                "AkArrayEnhancedWithOperators",
            )
        super().__init__(ak_array)

    def __eq__(self, x):
        if isinstance(x, (int, float)) and len(self.fields) > 0:
            return AkArrayEnhancedWithOperators({
                field: self[field].to_ak() == x for field in self.fields
            })
        else:
            try:
                return super().__add__(x)
            except:
                raise NotImplementedError

    def __ne__(self, x):
        return self.__eq__(x) == False

    def __lt__(self, x):
        for field in self.fields:
            print(self[field].to_ak())
            print(self[field].to_ak() < x)
        if isinstance(x, (int, float)) and len(self.fields) > 0:
            return AkArrayEnhancedWithOperators({
                field: self[field].to_ak() < x for field in self.fields
            })
        else:
            try:
                return super().__add__(x)
            except:
                raise NotImplementedError

    def __le__(self, x):
        for field in self.fields:
            print(self[field].to_ak())
            print(self[field].to_ak() < x)
        if isinstance(x, (int, float)) and len(self.fields) > 0:
            return AkArrayEnhancedWithOperators({
                field: self[field].to_ak() <= x for field in self.fields
            })
        else:
            try:
                return super().__add__(x)
            except:
                raise NotImplementedError

    def __gt__(self, x):
        if isinstance(x, (int, float)) and len(self.fields) > 0:
            return AkArrayEnhancedWithOperators({
                field: self[field].to_ak() > x for field in self.fields
            })
        else:
            try:
                return super().__add__(x)
            except:
                raise NotImplementedError

    def __ge__(self, x):
        if isinstance(x, (int, float)) and len(self.fields) > 0:
            return AkArrayEnhancedWithOperators({
                field: self[field].to_ak() >= x for field in self.fields
            })
        else:
            try:
                return super().__add__(x)
            except:
                raise NotImplementedError

    def __add__(self, x):
        if isinstance(x, (int, float)) and len(self.fields) > 0:
            return AkArrayEnhancedWithOperators({
                field: x + self[field].to_ak() for field in self.fields
            })
        elif isinstance(x, AkArrayEnhancedWithOperators):
            return AkArrayEnhancedWithOperators({
                field: x[field].to_ak() + self[field].to_ak() for field in self.fields
            })
        elif isinstance(x, ak.Array):
            return AkArrayEnhancedWithOperators({
                field: x[field] + self[field].to_ak() for field in self.fields
            })
        else:
            return super().__add__(x)

    def __radd__(self, x):
        return self + x

    def __sub__(self, x):
        return self.__add__(-x)

    def __rsub__(self, x):
        return x + self.__neg__()

    def __mul__(self, x):
        if isinstance(x, np.floating):
            x = x.item()
        if isinstance(x, (int, float)) and len(self.fields) > 0:
            return AkArrayEnhancedWithOperators({
                field: self[field].to_ak() * x for field in self.fields
            })
        elif isinstance(x, AkArrayEnhancedWithOperators):
            return AkArrayEnhancedWithOperators({
                field: x[field].to_ak() * self[field].to_ak() for field in self.fields
            })
        elif isinstance(x, ak.Array):
            return AkArrayEnhancedWithOperators({
                field: x[field] * self[field].to_ak() for field in self.fields
            })
        else:
            return super().__mul__(x)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __truediv__(self, x):
        return self.__mul__(1/x)

    def __neg__(self):
        return self.__mul__(-1)

    def __pow__(self, x):
        if isinstance(x, (int, float)) and len(self.fields) > 0:
            return AkArrayEnhancedWithOperators({
                field: self[field].to_ak() ** x for field in self.fields
            })
        else:
            return super().__pow__(x)

    # Black magic to do multiplication by numpy on the left
    # tinyurl.com/3795jb6b
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        lhs, rhs = inputs
        return rhs * lhs

    def __getattr__(self, attr):
        return AkArrayEnhancedWithOperators(super().__getattr__(attr))

    def __getitem__(self, item):
        return AkArrayEnhancedWithOperators(super().__getitem__(item))

