import os
import scipy
from pathlib import Path
from ssda.data.preprocess.porous import inpute_nans


if __name__=="__main__":
    data_path = Path("./data/")
    mat_path = list(data_path.glob("*.mat"))[0]
    mat = scipy.io.loadmat(mat_path,squeeze_me=True, struct_as_record=False)

    all_nds = mat["NDS"].transpose(2, 0, 1)
    number_of_examples = all_nds.shape[0]

    selected_simulations_data = {"nds": [],
                                 "PoDmD": [],
                                 "name": []}

    for example_index in range(number_of_examples):
        nds_example = all_nds[example_index]
        selected_simulations_data["nds"].append(inpute_nans(nds_example))
        selected_simulations_data["PoDmD"].append(mat["PoDmD"][example_index])
        selected_simulations_data["name"].append(example_index)