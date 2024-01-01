"""Run the default preprocessing pipeline on soarrKULee."""
import argparse
import datetime
import gzip
import json
import logging
import os
from typing import Any, Dict, Sequence

import librosa
import numpy as np
import math
import scipy.signal.windows
from brain_pipe.dataloaders.path import GlobLoader
from brain_pipe.pipeline.default import DefaultPipeline
from brain_pipe.preprocessing.brain.artifact import (
    InterpolateArtifacts,
    ArtifactRemovalMWF,
)
from brain_pipe.preprocessing.brain.eeg.biosemi import (
    biosemi_trigger_processing_fn,
)
from brain_pipe.preprocessing.brain.eeg.load import LoadEEGNumpy
from brain_pipe.preprocessing.brain.epochs import SplitEpochs
from brain_pipe.preprocessing.brain.link import (
    LinkStimulusToBrainResponse,
    BIDSStimulusInfoExtractor,
)
from brain_pipe.preprocessing.brain.rereference import CommonAverageRereference
from brain_pipe.preprocessing.brain.trigger import (
    AlignPeriodicBlockTriggers,
)
from brain_pipe.preprocessing.filter import SosFiltFilt
from brain_pipe.preprocessing.resample import ResamplePoly
from brain_pipe.preprocessing.stimulus.audio.envelope import GammatoneEnvelope
from brain_pipe.preprocessing.stimulus.audio.spectrogram import LibrosaMelSpectrogram

from brain_pipe.preprocessing.stimulus.load import LoadStimuli
from brain_pipe.runner.default import DefaultRunner
from brain_pipe.save.default import DefaultSave
# from mel import DefaultSave
from brain_pipe.utils.log import default_logging, DefaultFormatter
from brain_pipe.utils.path import BIDSStimulusGrouper


class BIDSAPRStimulusInfoExtractor(BIDSStimulusInfoExtractor):
    """Extract BIDS compliant stimulus information from an .apr file."""

    def __call__(self, brain_dict: Dict[str, Any]):
        """Extract BIDS compliant stimulus information from an events.tsv file.

        Parameters
        ----------
        brain_dict: Dict[str, Any]
            The data dict containing the brain data path.

        Returns
        -------
        Sequence[Dict[str, Any]]
            The extracted event information. Each dict contains the information
            of one row in the events.tsv file
        """
        event_info = super().__call__(brain_dict)
        # Find the apr file
        path = brain_dict[self.brain_path_key]
        apr_path = "_".join(path.split("_")[:-1]) + "_eeg.apr"
        # Read apr file
        apr_data = self.get_apr_data(apr_path)
        # Add apr data to event info
        for e_i in event_info:
            e_i.update(apr_data)
        #print("#############3apr data event_info########3",event_info)
        return event_info

    def get_apr_data(self, apr_path: str):
        """Get the SNR from an .apr file.

        Parameters
        ----------
        apr_path: str
            Path to the .apr file.

        Returns
        -------
        Dict[str, Any]
            The SNR.
        """
        import xml.etree.ElementTree as ET

        apr_data = {}
        tree = ET.parse(apr_path)
        root = tree.getroot()

        # Get SNR
        interactive_elements = root.findall(".//interactive/entry")
        for element in interactive_elements:
            description_element = element.find("description")
            if description_element.text == "SNR":
                apr_data["snr"] = element.find("new_value").text
        if "snr" not in apr_data:
            logging.warning(f"Could not find SNR in {apr_path}.")
            apr_data["snr"] = 100.0
        return apr_data


def default_librosa_load_fn(path):
    """Load a stimulus using librosa.

    Parameters
    ----------
    path: str
        Path to the audio file.

    Returns
    -------
    Dict[str, Any]
        The data and the sampling rate.
    """
    data, sr = librosa.load(path, sr=None)
    return {"data": data, "sr": sr}


def default_npz_load_fn(path):
    """Load a stimulus from a .npz file.

    Parameters
    ----------
    path: str
        Path to the .npz file.

    Returns
    -------
    Dict[str, Any]
        The data and the sampling rate.
    """
    np_data = np.load(path)
    return {
        "data": np_data["audio"],
        "sr": np_data["fs"],
    }


DEFAULT_LOAD_FNS = {
    ".wav": default_librosa_load_fn,
    ".mp3": default_librosa_load_fn,
    ".npz": default_npz_load_fn,
}


def temp_stimulus_load_fn(path):
    """Load stimuli from (Gzipped) files.

    Parameters
    ----------
    path: str
        Path to the stimulus file.

    Returns
    -------
    Dict[str, Any]
        Dict containing the data under the key "data" and the sampling rate
        under the key "sr".
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f_in:
            data = dict(np.load(f_in))
        return {
            "data": data["audio"],
            "sr": data["fs"],
        }

    extension = "." + ".".join(path.split(".")[1:])
    if extension not in DEFAULT_LOAD_FNS:
        raise ValueError(
            f"Can't find a load function for extension {extension}. "
            f"Available extensions are {str(list(DEFAULT_LOAD_FNS.keys()))}."
        )
    load_fn = DEFAULT_LOAD_FNS[extension]
    return load_fn(path)


def bids_filename_fn(data_dict, feature_name, set_name=None):
    """Default function to generate a filename for the data.

    Parameters
    ----------
    data_dict: Dict[str, Any]
        The data dict containing the data to save.
    feature_name: str
        The name of the feature.
    set_name: Optional[str]
        The name of the set. If no set name is given, the set name is not
        included in the filename.

    Returns
    -------
    str
        The filename.
    """

    filename = os.path.basename(data_dict["data_path"]).split("_eeg")[0]

    subject = filename.split("_")[0]
    session = filename.split("_")[1]
    filename += f"_desc-preproc-audio-{os.path.basename(data_dict.get('stimulus_path', '*.')).split('.')[0]}_{feature_name}"

    if set_name is not None:
        filename += f"_set-{set_name}"

    return os.path.join(subject, session, filename + ".npy")

def get_hop_length(arg, data_dict):
    return int((1 / 128) * data_dict["stimulus_sr"])
def get_n_fft(arg, data_dict):
    return int(math.pow(2, math.ceil(math.log2(int(0.025 * data_dict["stimulus_sr"])))))
def get_win_length(arg, data_dict):
    return int(0.025 * data_dict["stimulus_sr"])

def get_default_librosa_kwargs():

    librosa_kwargs = {
        "window": 'hann',
        "hop_length": get_hop_length,
        "n_fft": get_n_fft,
        "win_length": get_win_length,
        "fmin": 0,
        "fmax": 5000,
        "htk": False,
        "n_mels": 10,
        "center": False,
        "norm": 'slaney'
    }
    return librosa_kwargs

def run_preprocessing_pipeline(
        root_dir,
        preprocessed_stimuli_dir,
        preprocessed_eeg_dir,
        nb_processes=4,
        overwrite=False,
        log_path="sparrKULee.log",
):
    """Construct and run the preprocessing on SparrKULee.

    Parameters
    ----------
    root_dir: str
        The root directory of the dataset.
    preprocessed_stimuli_dir:
        The directory where the preprocessed stimuli should be saved.
    preprocessed_eeg_dir:
        The directory where the preprocessed EEG should be saved.
    nb_processes: int
        The number of processes to use. If -1, the number of processes is
        automatically determined.
    overwrite: bool
        Whether to overwrite existing files.
    log_path: str
        The path to the log file.
    """
    #########
    # PATHS #
    #########
    os.makedirs(preprocessed_eeg_dir, exist_ok=True)
    os.makedirs(preprocessed_stimuli_dir, exist_ok=True)

    ###########
    # LOGGING #
    ###########
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(DefaultFormatter())
    default_logging(handlers=[handler])

    ################
    # DATA LOADING #
    ################
    logging.info("Retrieving BIDS layout...")
    data_loader = GlobLoader(
        [os.path.join(root_dir, "sub-*", "*", "eeg", "*.bdf*")],
        filter_fns=[lambda x: "restingState" not in x],
        key="data_path",
    )

    #########
    # STEPS #
    #########

    stimulus_steps = DefaultPipeline(
        steps=[
            LoadStimuli(load_fn=temp_stimulus_load_fn),
            
            ResamplePoly(16000, data_key = ['stimulus_data'], sampling_frequency_key = ['stimulus_sr'], axis=0),
            DefaultSave(
               preprocessed_stimuli_dir,
               to_save={'wav': 'stimulus_data' },
               overwrite=overwrite
          ),
              
                      ],
        on_error=DefaultPipeline.RAISE,
        
          )

    eeg_steps = [
        LinkStimulusToBrainResponse(
            stimulus_data=stimulus_steps,
            extract_stimuli_information_fn=BIDSAPRStimulusInfoExtractor(),
            grouper=BIDSStimulusGrouper(
                bids_root=root_dir,
                mapping={"stim_file": "stimulus_path", "trigger_file": "trigger_path"},
                subfolders=["stimuli", "eeg"],
            ),
        ) 
    ]

    #########################
    # RUNNING THE PIPELINE  #
    #########################

    logging.info("Starting with the EEG preprocessing")
    logging.info("===================================")

    # Create data_dicts for the EEG files
    # Create the EEG pipeline
    eeg_pipeline = DefaultPipeline(steps=eeg_steps)

    DefaultRunner(
        nb_processes=nb_processes,
        logging_config=lambda: None,
    ).run(
        [(data_loader, eeg_pipeline)],

    )


if __name__ == "__main__":
    # Load the config
    # get the top folder of the dataset
    
    with open('config.json', "r") as f:
        config = json.load(f)

    # Set the correct paths as default arguments
    dataset_folder = config["dataset_folder"]
    derivatives_folder = os.path.join(dataset_folder, 'wav')
    preprocessed_stimuli_folder = os.path.join(
        derivatives_folder, config["preprocessed_stimuli_folder"]
    )
    preprocessed_eeg_folder = os.path.join(
        derivatives_folder, config["preprocessed_eeg_folder"]
    )
    default_log_folder = os.path.dirname(os.path.abspath(__file__))

    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description="Preprocess the auditory EEG dataset")
    parser.add_argument(
        "--nb_processes",
        type=int,
        default=8,
        help="Number of processes to use for the preprocessing. "
             "The default is to use all available cores (-1).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--log_path", type=str, default=os.path.join(
            default_log_folder,
            "sparrKULee_{datetime}.log"
        )
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default=dataset_folder,
        help="Path to the folder where the dataset is downloaded",
    )
    parser.add_argument(
        "--preprocessed_stimuli_path",
        type=str,
        default=preprocessed_stimuli_folder,
        help="Path to the folder where the preprocessed stimuli will be saved",
    )
    parser.add_argument(
        "--preprocessed_eeg_path",
        type=str,
        default=preprocessed_eeg_folder,
        help="Path to the folder where the preprocessed EEG will be saved",
    )
    args = parser.parse_args()

    # Run the preprocessing pipeline
    run_preprocessing_pipeline(
        args.dataset_folder,
        args.preprocessed_stimuli_path,
        args.preprocessed_eeg_path,
        args.nb_processes,
        args.overwrite,
        args.log_path.format(
            datetime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
    )
