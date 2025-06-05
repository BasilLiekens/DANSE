from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
from .parameters import sweepParameters


@dataclass
class parameterGenerator:
    """
    Class that allows to easily generate an iterator containing the parameters
    for that iteration of the parameter sweep.

    Each iteration of this generator returns a "sweepParameters" struct with
    some entries altered as dictated by "sweepData".

    The i'th element that will be returned will run over the parameters in
    reverse: i.e. the parameter that changes fastest is the one that was passed
    in last.

    Example:
    >>> sweepData = [("A", [1, 2, 3]), ("B", [1, 2])]
        results in the following sequence:
    >>> "A" = 1, "B" = 1
    >>> "A" = 1, "B" = 2
    >>> "A" = 2, "B" = 1
    >>> ...

    Parameters
    ----------
    parameters: sweepParameters
        The basic parameters to start from and alter during the sweeps

    sweepData: list[tuple[str, list]]
        A list containing tuples with as first entry the name of the field to
        adjust and second entry a list with all values to sweep over

    coupling: list[list[str]]
        A list with each entry a list of the entries to couple (i.e. their
        values should be changed at the same time). This slightly changes the
        way in which the sweeping works: now all coupled parameters are adjusted
        at the same speed as the first encountered variable in the coupling.
        I.e., if the coupled variables are the first and last entries of
        "sweepData", the last value will also change the slowest, despite being
        in last position.
    """

    #
    parameters: sweepParameters = field(default_factory=sweepParameters())
    #
    sweepData: list[tuple[str, list]] = field(default_factory=lambda: [])
    coupling: list[list[str]] = field(default_factory=lambda: [])
    #

    def __post_init__(self):
        """
        Account for the coupling by changing the data structures slightly:
        regroup the sweepData to have lists of the coupled variables.
        """
        # sanity checks
        for tup in self.sweepData:
            if len(tup) != 2:
                raise ValueError("Expected all tuples to contain just two elements")
            if not isinstance(tup[0], str):
                raise ValueError(f"Expected first element of tuple to be a string")
            if not isinstance(tup[1], list):
                raise ValueError(f"Expected second element to be a list")

        self.values: list[list[tuple[str, list]]] = []
        # iterate over sweepData and take out all data in the list until the
        # list is empty (signifying that there are no variables left and all
        # data has been regrouped)
        while len(self.sweepData) > 0:
            data = [self.sweepData.pop(0)]  # remove the first element and relocate
            # iterate over the coupling to see if there's a match
            for i in range(len(self.coupling)):
                if data[0][0] in self.coupling[i]:
                    vars = self.coupling[i]
                    # ugly sequence to extract all indices of variables in
                    # "sweepData" that are in this set of coupling.
                    idcs = [
                        i
                        for i, x in enumerate(
                            [
                                self.sweepData[idx][0] in vars
                                for idx in range(len(self.sweepData))
                            ]
                        )
                        if x
                    ]
                    # relocate entries and pop: backwards to avoid issues with idcs
                    idcs.reverse()
                    for idx in idcs:
                        tmp = self.sweepData.pop(idx)
                        if len(tmp[1]) != len(data[0][1]):
                            raise ValueError(
                                "Coupled variables have different lengths!"
                            )
                        data.append(tmp)

            self.values.append(data)

    def __len__(self) -> int:
        lengths = np.prod([len(self.values[x][0][1]) for x in range(len(self.values))])
        return np.maximum(lengths, 1)  # edgecase with no parameters to sweep

    def __get_item__(self, idx: int) -> sweepParameters:
        params = deepcopy(self.parameters)

        for i in range(len(self.values) - 1, -1, -1):  # run backwards
            newParams = self.values[i]
            numParams = len(newParams[0][1])
            paramIdx = idx % numParams  # select tuple, then the list
            for j in range(len(newParams)):  # iterate over coupled variables
                param = newParams[j]
                params.__setattr__(param[0], param[1][paramIdx])
            # account for the seen variable => next index will change more slowly
            idx = idx // len(newParams[0][1])

        return params

    def __call__(self):
        if self.values == []:  # edge case with only one parameter
            return self.parameters
        else:
            for idx in range(self.__len__()):
                yield self.__get_item__(idx)

                if idx == self.__len__() - 1:
                    return
