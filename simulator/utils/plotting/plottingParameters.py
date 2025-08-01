def getParameters() -> tuple[dict[str, str], dict[str, str]]:
    """
    Utility function that provides the colors and markers (in that order) for
    plots of the progression of the DANSE algorithm. The results are provided as
    dictionaries with the keys being the title of the algorithm and the values
    being the actual values.
    """
    colors: dict[str, str] = {
        r"$DANSE_1$": "tab:blue",
        r"$rS-DANSE_1$": "tab:orange",
        r"$GEVD-DANSE_1$": "tab:green",
        r"$rS-GEVD-DANSE_1$": "tab:red",
    }

    markers: dict[str, str] = {
        r"$DANSE_1$": "o",
        r"$rS-DANSE_1$": "x",
        r"$GEVD-DANSE_1$": "+",
        r"$rS-GEVD-DANSE_1$": "v",
    }

    return colors, markers
