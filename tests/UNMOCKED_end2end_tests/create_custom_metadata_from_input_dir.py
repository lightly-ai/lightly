import sys

from lightly.data import LightlyDataset
from lightly.utils import save_custom_metadata

if __name__ == "__main__":
    if len(sys.argv) == 1 + 2:
        input_dir, metadata_filename= \
            (sys.argv[1 + i] for i in range(2))
    else:
        raise ValueError("ERROR in number of command line arguments, must be 2."
                         "Example: python create_custom_metadata_from_input_dir input_dir metadata_filename")

    dataset = LightlyDataset(input_dir)

    # create a list of pairs of (filename, metadata)
    custom_metadata = []
    for index, filename in enumerate(dataset.get_filenames()):
        metadata = {'index': index}
        custom_metadata.append((filename, metadata))

    save_custom_metadata(metadata_filename, custom_metadata)


