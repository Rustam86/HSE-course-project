import json
import logging
import os
import random
import sys
import subprocess
import tempfile
import time
from typing import Dict, List, Tuple, Callable, Optional, Union
import warnings

from ete3 import ClusterTree
import colorsys
from google.colab import files
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy
import seaborn as sns
import torch
import Bio
from Bio import Entrez, SeqIO
from Bio.SeqUtils import gc_fraction, GC_skew, MeltingTemp as mt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, is_valid_linkage
from scipy import ndimage, stats
import sklearn
from sklearn.linear_model import LinearRegression
import torch
from torch import nn
import tqdm as tq
from tqdm import tqdm
import transformers
from transformers import BertForTokenClassification, BertTokenizer

# Set email (required for accessing NCBI databases via Entrez)
Entrez.email = "rustam@heydarov.ru"

ZDNABERT_PARAMS = {
    'model_confidence_threshold': 0.2,
    'minimum_sequence_length': 10,
    'tokenizer': BertTokenizer.from_pretrained('/content/6-new-12w-0/'),
    'model': BertForTokenClassification.from_pretrained('/content/6-new-12w-0/')
}

ZHUNT_PARAMS = {
    'zhunt_path': '/content/zhunt3',
    'score': 500,
    'window_size': 6,
    'min_size': 3,
    'max_size': 6
}

colors = ['#003f5c', '#ffb5a7', '#9c88ff', '#ff7b00', '#955196', '#b5838d', '#edc7b7', '#6b705c',
          '#f4d03f', '#d4af37', '#da627d', '#ff6e54', '#d1ccc0', '#7c7c7c', '#b7c0c7', '#c4aead',
          '#8d99ae', '#444e86', '#9d8189', '#6497b1', '#d6e2e9', '#e5989b', '#006d77', '#95afc0',
          '#dd5182', '#ffa600', '#5a189a', '#2e3440',  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7c1e30', '#b29e7c',
          '#2d2d87', '#beaed4', '#fdc086', '#bf5b17', '#f0027f', '#fbb4ae', '#fed9a6', '#b3cde3']


PREVALENT_HOSTS = {
    'Alphacoronavirus_1': 'Cats',
    'Alphacoronavirus_AMALF': 'Bats',
    'Alphacoronavirus_BT020': 'Bats',
    'Alphacoronavirus_CHB25': 'Bats',
    'Alphacoronavirus_HKU33': 'Bats',
    'Alphacoronavirus_WA1087': 'Bats',
    'Alphacoronavirus_WA2028': 'Bats',
    'Alphacoronavirus_WA3607': 'Bats',
    'Alphapironavirus_bona': 'Salmon',
    'Avian_coronavirus': 'Chicken',
    'Avian_coronavirus_9203': 'Chicken',
    'Bat_Hp-betacoronavirus_Zhejiang2013': 'Bats',
    'Bat_coronavirus_CDPHE15': 'Bats',
    'Bat_coronavirus_HKU10': 'Bats',
    'Beluga_whale_coronavirus_SW1': 'Whale',
    'Betacoronavirus_1': 'Human/Swine/Dog',
    'Bulbul_coronavirus_HKU11': 'Birds',
    'China_Rattus_coronavirus_HKU24': 'Rat',
    'Common_moorhen_coronavirus_HKU21': 'Birds',
    'Coronavirus_HKU15': 'Swine',
    'Duck_coronavirus_2714': 'Birds',
    'Eidolon_bat_coronavirus_C704': 'Bats',
    'Goose_coronavirus_CB17': 'Birds',
    'Hedgehog_coronavirus_1': 'Hedgehog',
    'Human_coronavirus_229E': 'Human',
    'Human_coronavirus_HKU1': 'Human',
    'Human_coronavirus_NL63': 'Human',
    'Human_coronavirus_OC43': 'Human',
    'Lucheng_Rn_rat_coronavirus': 'Rat',
    'Middle_East_respiratory_syndrome-related_coronavirus': 'Human',
    'Miniopterus_bat_coronavirus_1': 'Bats',
    'Miniopterus_bat_coronavirus_HKU8': 'Bats',
    'Mink_coronavirus_1': 'Mink',
    'Munia_coronavirus_HKU13': 'Birds',
    'Murine_coronavirus': 'Mice',
    'Myodes_coronavirus_2JL14': 'Myodes',
    'Myotis_ricketti_alphacoronavirus_Sax-2011': 'Bats',
    'NL63-related_bat_coronavirus_strain_BtKYNL63-9b': 'Bats',
    'Night_heron_coronavirus_HKU19': 'Birds',
    'Nyctalus_velutinus_alphacoronavirus_SC-2013': 'Bats',
    'Pipistrellus_bat_coronavirus_HKU5': 'Bats',
    'Pipistrellus_kuhlii_coronavirus_3398': 'Bats',
    'Porcine_epidemic_diarrhea_virus': 'Swine',
    'Rhinolophus_bat_coronavirus_HKU2': 'Bats',
    'Rhinolophus_ferrumequinum_alphacoronavirus_HuB-2013': 'Bats',
    'Rousettus_bat_coronavirus_GCCDC1': 'Bats',
    'Rousettus_bat_coronavirus_HKU9': 'Bats',
    'Scotophilus_bat_coronavirus_512': 'Bats',
    'Severe_acute_respiratory_syndrome_related_coronavirus': 'Human',
    'Severe_acute_respiratory_syndrome_related_coronavirus_2': 'Human',
    'Sorex_araneus_coronavirus_T14': 'Shrew',
    'Suncus_murinus_coronavirus_X74': 'Shrew',
    'Tylonycteris_bat_coronavirus_HKU4': 'Bats',
    'White-eye_coronavirus_HKU16': 'Birds',
    'Wigeon_coronavirus_HKU20': 'Birds'
}


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that converts numpy types to standard Python types.

    Extends the standard JSON encoder class to handle numpy integer, floating-point,
    and array types, converting them to standard Python types.
    """

    def default(self, obj):
        """
        Override the default method to handle numpy types.

        Parameters:
        obj: Object to encode.

        Returns:
        Encoded object.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def sum_intervals(intervals: List[Tuple[int, int]]) -> int:
    """
    Calculate the total sum of the lengths of given intervals, accounting for overlaps.

    Parameters:
    intervals (List[Tuple[int, int]]): List of intervals represented as tuples of start and end.

    Returns:
    int: Total length of all intervals, adjusted for overlaps. Returns 0 if the input list is empty.
    """
    if not intervals:
        return 0

    # Sort intervals based on start times
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return sum([end - start for start, end in merged])

def save_json(data: dict, filename: str) -> None:
    """
    Save data to a JSON file.

    Parameters:
    - data (dict): The Python dictionary to save.
    - filename (str): The name of the file where the data should be saved.

    Returns:
    None
    """
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4, cls=NumpyEncoder)

def read_json_file(file_path: str) -> Optional[dict]:
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except PermissionError:
        print(f"Permission denied for file: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None

def list_files(directory: str) -> List[str]:
    try:
        return [os.path.join(directory, filename)
                for filename in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, filename))]
    except PermissionError:
        print(f"Permission denied for directory: {directory}")
        return []


def datetime_to_ordinal(dt):
    if pd.notna(dt):  # check if the value is not NaT
        return dt.toordinal()
    else:
        return None

def filter_arrays(arrays):
    """
    Filters arrays by removing those that contain NaN values or have a length of zero.

    Parameters:
    - arrays (list): A list of arrays.

    Returns:
    - filtered_arrays (list): A list of filtered arrays.
    """
    filtered_arrays = []
    for array in arrays:
        if np.isnan(array).any() or len(array) == 0:
            continue
        filtered_arrays.append(array)
    return filtered_arrays

def get_high_contrast_colors(n):
    HSV_tuples = [(x*1.0/n, 0.8, 0.9) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    hex_colors = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r,g,b in RGB_tuples]
    return hex_colors

def generate_high_contrast_colors(n):
    colors = []
    for i in range(n):
        # Cycle through hue
        hue = float(i) / n
        # Alternate between full and half brightness
        lightness = 0.5 if i % 2 == 0 else 0.8
        # Keep saturation constant
        saturation = 0.9
        # Convert to RGB
        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(hue, lightness, saturation)]
        colors.append(f'#{r:02x}{g:02x}{b:02x}')
    return colors

def get_high_contrast_colors_gr(n):
    phi = 0.618033988749895
    h = random.random()  # random start value
    colors = []
    for _ in range(n):
        h += phi
        h %= 1
        colors.append(mcolors.hsv_to_rgb([h, 1, 1]))
    return colors

unique_values = [...]  # Your list of unique values
colors = get_high_contrast_colors(len(unique_values))
color_map = dict(zip(unique_values, colors))


def fetch_gb_files(accession_list: List[str], email: str, output_dir: str = ".") -> None:
    """
    Fetch GB (GenBank) files for a list of accession numbers.

    Parameters:
    accession_list (List[str]): List of accession numbers.
    email (str): Your email address. NCBI requires this to track usage.
    output_dir (str): Directory to save the GB files. Default is the current directory.

    Returns:
    None. GB files are saved to the specified directory.
    """

    # Set email for Entrez
    Entrez.email = email

    # Iterate through the list of accession numbers
    for accession in accession_list:
        # Fetch the data for the given accession number
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")

        # Form the filename where the data will be saved
        filename = f"{output_dir}/{accession}.gb"

        # Write the data to the file
        with open(filename, "w") as f:
            f.write(handle.read())

        # Close the handle
        handle.close()

        print(f"{accession}.gb saved to {output_dir}")

def list_proteins(gb_file: str) -> List[str]:
    """
    List all protein names from a GB (GenBank) file.

    Parameters:
    gb_file (str): Path to the GenBank file.

    Returns:
    List[str]: List of protein names.
    """

    protein_names = []

    # Parse the GenBank file
    for record in SeqIO.parse(gb_file, "genbank"):
        for feature in record.features:
            # Check if the feature is a coding sequence (CDS) and has a product
            if feature.type == "CDS" and "product" in feature.qualifiers:
                protein_names.append(feature.qualifiers["product"][0])

    return protein_names

def extract_protein_sequence(gb_file: str, protein_name: str) -> Optional[str]:
    """
    Extract amino acid sequence from a GB (GenBank) file given the name of a protein.

    Parameters:
    gb_file (str): Path to the GenBank file.
    protein_name (str): Name of the protein to extract the sequence for.

    Returns:
    str, None: Amino acid sequence as a string if found, else None.
    """

    # Parse the GenBank file
    for record in SeqIO.parse(gb_file, "genbank"):
        for feature in record.features:
            if feature.type == "CDS":
                # Check if the protein product name matches the given name
                if "product" in feature.qualifiers and feature.qualifiers["product"][0] == protein_name:
                    # Extract the protein sequence
                    return feature.qualifiers["translation"][0]

    return None

def write_to_fasta(seq_names: List[str], sequences: List[str], filename: str) -> None:
    """
    Write sequences to a FASTA file.

    Parameters:
    seq_names (List[str]): List of sequence names.
    sequences (List[str]): List of corresponding sequences.
    filename (str): Name of the output FASTA file.

    Returns:
    None. Writes the sequences to the specified FASTA file.
    """

    with open(filename, 'w') as f:
        for name, seq in zip(seq_names, sequences):
            # Write the name and sequence to the file in FASTA format
            f.write(f">{name}\n{seq}\n")

# Generic function to run sequence processing functions on a batch of sequences
def process_fasta_with_function(fasta_path: str, function: Callable, params: Dict) -> Dict[str, List[Tuple[int, int]]]:
    """
    Run a function on a multi-fasta file.

    Parameters:
    fasta_path (str): The path to the fasta file.
    function (Callable): The function to run on each sequence.
    params (Dict): A dictionary containing the parameters for the function.

    Returns:
    Dict[str, List[Tuple[int, int]]]: A dictionary where the keys are the sequence
                                      identifiers and the values are the results of
                                      the function for each sequence.
    """
    results = {}

    # Load all the sequences into a list to compute the total number of sequences
    sequences = list(SeqIO.parse(fasta_path, "fasta"))

    # Wrap the sequences list with tqdm for progress bar
    with tqdm(sequences, desc="Processing", unit="sequence") as pbar:
        for record in pbar:
            sequence_id = record.id
            sequence = str(record.seq)
            results[sequence_id] = function(sequence, **params)

    return results

# Function to predict Z-RNA regions with ZDNABERT
def run_zdnabert(seq_string: str, model:
                 torch.nn.Module, tokenizer: torch.nn.Module,
                 model_confidence_threshold: float = 0.2,
                 minimum_sequence_length: int = 10) -> List[Tuple[int, int]]:
    """
    Process a DNA or RNA sequence string using a given model and identify segments of significance.

    Parameters:
    - seq_string (str): The DNA or RNA sequence to be processed.
    - model (torch.nn.Module): Pretrained model for sequence prediction.
    - tokenizer (torch.nn.Module): Tokenizer to convert the sequence into model-compatible tokens.
    - model_confidence_threshold (float, optional): Threshold for considering a segment significant. Defaults to 0.2.
    - minimum_sequence_length (int, optional): Minimum length of a segment to consider. Defaults to 10.

    Returns:
    List[Tuple[int, int]]: A list of tuples representing the start and end positions
    of the predicted Z-DNA forming regions.
    """

    model.cuda()
    # Convert the sequence to k-mers
    k = 6
    kmer_seq = [seq_string.upper()[x:x+k] for x in range(len(seq_string)+1-k)]

    # Split the kmer sequence into pieces of a specified length with some overlap (padding)
    length, pad = 512, 16
    seq_pieces = [kmer_seq[st:min(st+512, len(kmer_seq))] for st in range(0, len(kmer_seq), length-pad)]

    # Use the model to predict on each sequence piece
    preds = []
    with torch.no_grad():
        for seq_piece in seq_pieces:
            input_ids = torch.LongTensor(tokenizer.encode(' '.join(seq_piece), add_special_tokens=False))
            outputs = torch.softmax(model(input_ids.cuda().unsqueeze(0))[-1], axis=-1)[0, :, 1]
            preds.append(outputs.cpu().numpy())

    # Stitch together the predictions for each piece of the sequence
    res = np.array([])
    for seq in preds:
        res = np.concatenate([res[:-pad], seq])
    stitched_seq = res

    # Identify segments with prediction confidence above the threshold
    out = []
    labeled, max_label = scipy.ndimage.label(stitched_seq > model_confidence_threshold)
    for label in range(1, max_label+1):
        candidate = np.where(labeled == label)[0]
        candidate_length = candidate.shape[0]
        # Consider segments only if they are longer than the specified minimum sequence length
        if candidate_length > minimum_sequence_length:
            out.append((candidate[0], candidate[-1]))

    return out

# Function to predict Z-RNA regions with ZHUNT
def run_zhunt(seq_string: str, zhunt_path: str, score: int = 500,
              window_size: int = 6, min_size: int = 3,
              max_size: int = 6) -> List[Tuple[int, int]]:
    """
    Run the ZHunt program to predict Z-DNA forming regions in a DNA sequence.

    Parameters:
    - seq_string (str): The DNA or RNA sequence to be processed.
    - zhunt_path (str): The path to the ZHunt executable.
    - window_size (int): The window size for the ZHunt program. Default is 6.
    - min_size (int): The minimum size for the ZHunt program. Default is 3.
    - max_size (int): The maximum size for the ZHunt program. Default is 6.

    Returns:
    List[Tuple[int, int]]: A list of tuples representing the start and end positions
    of the predicted Z-DNA forming regions.
    """
    # Ensure the sequence only contains valid DNA bases
    # assert set(sequence).issubset({"A", "C", "G", "T", "N"}), "Invalid DNA sequence"

    # Create a temporary file
    file_descriptor, temp_file_path = tempfile.mkstemp()
    os.close(file_descriptor)

    try:
        # Write the sequence to the temporary file
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(seq_string)

        # Run the ZHunt program
        subprocess.run(
            [zhunt_path, str(window_size), str(min_size), str(max_size), temp_file_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            input=seq_string, encoding='ascii'
        )

        # Read the ZHunt output into a DataFrame
        with open(temp_file_path + ".Z-SCORE", 'r') as zhunt_output:
            output_data = pd.read_csv(zhunt_output,
                             names=['Start', 'End', 'nu-1', 'nu-2', 'nu-3',
                                    'ZH-Score', 'Sequence', 'Conformation'],
                             skiprows=1, sep='\s+')

        # Filter the DataFrame to only include rows with a ZH-Score greater than 500
        filtered_data = output_data[output_data['ZH-Score'] > score]

        # Return a list of tuples representing the start and end positions of the predicted Z-DNA forming regions
        return list(zip(filtered_data['Start'], filtered_data['End']))

    except Exception as error:
        print(f"An error occurred while running ZHunt: {error}")

    finally:
        # Clean up the temporary files
        os.remove(temp_file_path)
        os.remove(temp_file_path + ".Z-SCORE")

def get_taxid(species_name: str) -> Optional[str]:
    """
    Retrieve the taxonomic identifier (taxid) for a given species name.

    Args:
        species_name (str): The name of the species for which the taxid is required.

    Returns:
        str: The taxid for the species, if found. None otherwise.
    """
    handle = Entrez.esearch(db="taxonomy", term=species_name)
    record = Entrez.read(handle)
    handle.close()
    try:
        return record["IdList"][0]
    except IndexError:
        return None

def fetch_genbank_taxid(taxid: str, filename: str, n: int) -> None:
    """
    Download a fixed number of random genomic GenBank files for a given taxid and save to a file.

    Args:
        taxid (str): The taxid for which GenBank files are required.
        filename (str): The name of the file to save the GenBank data.
        n (int): The fixed number of GenBank files to fetch.

    Returns:
        None
    """
    term = f'txid{taxid}[organism:exp] AND biomol_genomic[prop] AND ("25000"[SLEN] : "35000"[SLEN])'
    handle = Entrez.esearch(db="nuccore",
                            term=term,
                            retmax=10000000)
    record = Entrez.read(handle)
    id_list = record["IdList"]

    if not id_list:
        print(f"No GenBank records found for taxid {taxid}")
        return

    # Randomly select n IDs from id_list
    selected_ids = random.sample(id_list, min(n, len(id_list)))

    # Using tqdm to show progress bar
    with open(filename, "w") as out, tqdm(total=len(selected_ids), desc="Fetching GenBank records") as pbar:
        for gb_id in selected_ids:
            handle = Entrez.efetch(db="nuccore", id=gb_id, rettype="gb", retmode="text")
            out.write(handle.read())
            pbar.update(1)  # update progress bar for each GenBank record
            time.sleep(0.35)  # Optional: to avoid hitting API rate limits

def count_genbank_entries(taxid: str) -> int:
    """
    Count the number of GenBank entries for a given taxid.

    Args:
        taxid (str): The taxid for which the GenBank entry count is required.

    Returns:
        int: The count of GenBank entries associated with the taxid.
    """
    term = f'txid{taxid}[organism:exp] AND biomol_genomic[prop] AND ("25000"[SLEN] : "35000"[SLEN])'
    handle = Entrez.esearch(db="nuccore",
                            term=term,
                            retmax=100000000)
    record = Entrez.read(handle)
    return len(record["IdList"])

def newick_to_linkage(newick: str, label_order: list[str] = None) -> (np.ndarray, list[str]):
    """
    Convert a Newick formatted tree into a linkage matrix and retrieve corresponding labels.

    Parameters:
    - newick (str): A Newick formatted string representation of the tree.
    - label_order (list[str], optional): Desired order of labels in the output.
                                         If not provided, the order from the Newick string is used.

    Returns:
    - np.ndarray: Linkage matrix representation of the tree.
    - list[str]: List of labels corresponding to the tree nodes.

    Raises:
    - AssertionError: If there are labels in `label_order` which are not present in the Newick string.
    """

    # Convert Newick string to ClusterTree
    tree = ClusterTree(newick)

    # Get the cophenetic matrix and labels from the tree
    cophenetic_matrix, newick_labels = tree.cophenetic_matrix()
    cophenetic_matrix = pd.DataFrame(cophenetic_matrix, columns=newick_labels, index=newick_labels)

    # If a label order is provided, reorder the cophenetic matrix rows and columns accordingly
    if label_order is not None:
        # Identify missing and superfluous labels
        missing_labels = set(label_order).difference(set(newick_labels))
        superfluous_labels = set(newick_labels).difference(set(label_order))

        # Check for labels that are in `label_order` but not in Newick string
        assert len(missing_labels) == 0, f'Some labels are not in the newick string: {missing_labels}'

        # Warn if there are labels in the Newick string that are not used in `label_order`
        if len(superfluous_labels) > 0:
            logging.warning(f'Newick string contains unused labels: {superfluous_labels}')

        # Reorder the cophenetic matrix
        cophenetic_matrix = cophenetic_matrix.reindex(index=label_order, columns=label_order)

    # Convert the cophenetic matrix to a pairwise distance matrix
    pairwise_distances = squareform(cophenetic_matrix)

    # Return linkage matrix and labels
    return linkage(pairwise_distances), list(cophenetic_matrix.columns)

def total_interval_length(intervals_dict: Dict[str, List[Tuple[int, int]]]) -> Dict[str, int]:
    """
    Given a dictionary of intervals for each ID, this function calculates
    the total length of the intervals for each ID.

    Parameters:
    intervals_dict (Dict[str, List[Tuple[int, int]]]): Dictionary where key is ID and value is a list of intervals.

    Returns:
    Dict[str, int]: Dictionary where key is ID and value is total length of intervals.
    """

    length_dict = {}

    for key, intervals in intervals_dict.items():
        # Calculate total length for the current key by subtracting start from end for each interval
        total_length = sum([end - start for start, end in intervals])
        # Splitting the key and using the first part as the new key
        length_dict[key.split('.')[0]] = total_length

    return length_dict

def plot_tree_with_annotations(newick: str, numeric_values: dict,
                               taxa_db: pd.DataFrame, color_threshold: float = 0.9) -> None:
    """
    Plot a tree with annotations using a Newick formatted string along with numeric values
    represented as a bar plot beside the dendrogram.

    Parameters:
    - newick (str): A Newick formatted string representation of the tree.
    - numeric_values (dict): A dictionary containing the numeric values corresponding to the labels. e.g. {"label1": 5.6}
    - taxa_db (pd.DataFrame): A DataFrame containing 'Accession' and 'Prevalent host' data.
    - color_threshold (float, optional): The threshold for coloring branches. Defaults to 0.9.

    Returns:
    - None
    """

    # Convert newick string to linkage matrix and get labels
    linkage_matrix, labels = newick_to_linkage(newick)  # Ensure you've defined newick_to_linkage function

    switch = False

    if switch:
      labels = [dict(zip((taxa_db['Accession']), taxa_db['Species']))[id] for id in labels]

    # Create a triple subplot - one for the tree, one for numeric value bars, and one for the legend
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 0.5, 0.5]})

    # Plot tree on the left subplot
    dendro_data = dendrogram(linkage_matrix, labels=labels, orientation='left', ax=ax1, color_threshold=color_threshold)

    # Generate host-related color mapping from the DataFrame
    host_groups = sorted(set(taxa_db['Prevalent host'].unique()))
    id_to_host = dict(zip(taxa_db['Accession'], taxa_db['Prevalent host']))
    palette = sns.color_palette("husl", len(host_groups))
    host_to_color = dict(zip(host_groups, palette))

    # Update the color for each label according to its prevalent host
    label_color_mapping = {}
    for label in dendro_data['ivl']:
        if switch:
            label = dict(zip((taxa_db['Species']), taxa_db['Accession']))[label]
        prevalent_host = id_to_host.get(label)
        label_color_mapping[label] = host_to_color.get(prevalent_host, "grey")


    # Plot tree on the left subplot
    dendro_data = dendrogram(linkage_matrix, labels=labels, orientation='left', ax=ax1, color_threshold=color_threshold)

    # Get the number of leaves (labels) in tree
    num_leaves = len(dendro_data['ivl'])

    # Get the maximum numeric value for x-limit of the middle subplot
    max_width = max(numeric_values.values())

    # Plot numeric values as colored boxes in the middle subplot
    for label, y in zip(dendro_data['ivl'], range(num_leaves)):
        if switch:
            label = dict(zip((taxa_db['Species']), taxa_db['Accession']))[label]
        box_width = numeric_values.get(label, 0)
        color = label_color_mapping.get(label, "grey")
        ax2.add_patch(plt.Rectangle((0, y*(ax1.get_ylim()[1]/num_leaves) + 0.25*ax1.get_ylim()[1]/num_leaves),
                                    box_width, 8, facecolor=color))

    # Set limits for the middle subplot and hide its y-axis
    ax2.set_xlim(0, max_width)
    ax2.set_ylim(ax1.get_ylim())
    ax2.yaxis.set_visible(False)

    # Hide x and y axes for the legend axis
    ax3.axis('off')

    # Populate the legend on the third axis
    for i, host in enumerate(host_groups):
        ax3.plot([], [], 'o', color=host_to_color[host], label=host)
    ax3.legend(title='Prevalent Host', loc='center')

    plt.tight_layout()
    plt.show()

def compute_jaccard_index(set_intervals1: List[Tuple[int, int]], set_intervals2: List[Tuple[int, int]]) -> float:
    """
    This function computes the Jaccard index between two sets of intervals.

    Parameters:
    - set_intervals1, set_intervals2 (list of tuples): The intervals to be compared.

    Returns:
    jaccard_index (float): The Jaccard index between the two sets of intervals.
    """

    # Compute the intersection of the two sets of intervals
    intersection_size = sum(min(interval2[1], interval1[1]) - max(interval2[0], interval1[0])
                            for interval1 in set_intervals1
                            for interval2 in set_intervals2
                            if interval1[1] > interval2[0] and interval2[1] > interval1[0])

    # Compute the size of intervals in set_intervals1 that do not overlap with set_intervals2
    non_overlap_size1 = sum(interval1[1] - interval1[0]
                            for interval1 in set_intervals1
                            if not any(interval1[1] > interval2[0] and interval2[1] > interval1[0]
                                       for interval2 in set_intervals2))

    # Compute the size of intervals in set_intervals2 that do not overlap with set_intervals1
    non_overlap_size2 = sum(interval2[1] - interval2[0]
                            for interval2 in set_intervals2
                            if not any(interval2[1] > interval1[0] and interval1[1] > interval2[0]
                                       for interval1 in set_intervals1))

    # Compute the size of the union of the two sets of intervals
    union_size = intersection_size + non_overlap_size1 + non_overlap_size2

    # Compute the Jaccard index
    if union_size:
        jaccard_index = intersection_size / union_size
    else:
        jaccard_index = 1

    return jaccard_index


def create_clustered_dataframe(zrna_intervals_dict: Dict) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    This function creates a clustered dataframe based on Jaccard indices of intervals.

    Parameters:
    - zrna_intervals_dict (Dict): The dictionary containing Z-RNA interval data.

    Returns:
    clustered_dataframe (pd.DataFrame): The clustered dataframe.
    row_linkage_matrix (np.ndarray): The hierarchical clustering encoded as a linkage matrix.
    """

    # Parse the intervals file
    intervals_dict = zrna_intervals_dict

    sequence_ids = list(intervals_dict.keys())

    # Check if there are enough sequences
    if len(sequence_ids) < 2:
        return pd.DataFrame({list(zrna_intervals_dict.keys())[0]: 0},
                            index=list(zrna_intervals_dict.keys()),
                            columns=list(zrna_intervals_dict.keys())), None, None

    # Compute the Jaccard index matrix with a single progress bar
    print("Computing Jaccard index matrix...")
    intervals_list = list(intervals_dict.items())
    interval_count = len(intervals_list)
    sequence_labels = [intervals_list[i][0] for i in range(interval_count)]

    jaccard_index_matrix = np.zeros((interval_count, interval_count))
    with tqdm(total=interval_count**2, desc="Computing Jaccard indices") as pbar:
        for i in range(interval_count):
            for j in range(interval_count):
                jaccard_index_matrix[i][j] = compute_jaccard_index(intervals_list[i][1], intervals_list[j][1])
                pbar.update(1)

    jaccard_dataframe = pd.DataFrame(jaccard_index_matrix, index=sequence_labels, columns=sequence_labels)

    # Perform hierarchical clustering on rows and columns
    print("Performing hierarchical clustering on rows...")
    row_linkage_matrix = linkage(jaccard_dataframe.values, method='average', metric='euclidean')
    print("Performing hierarchical clustering on columns...")
    column_linkage_matrix = linkage(jaccard_dataframe.values.T, method='average', metric='euclidean')

    # Reorder the dataframe based on the clustering
    print("Reordering dataframe based on clustering...")
    row_dendrogram = dendrogram(row_linkage_matrix, no_plot=True)
    column_dendrogram = dendrogram(column_linkage_matrix, no_plot=True)
    clustered_dataframe = jaccard_dataframe.iloc[row_dendrogram['leaves'], column_dendrogram['leaves']]

    return clustered_dataframe, row_linkage_matrix

def extract_regions_from_genbank_id(genbank_id: str) -> List[Tuple[str, int, int]]:
    """
    Extract genomic regions including CDS (excluding "ORF1ab polyprotein" and "ORF1a polyprotein"),
    mat_peptides, and UTRs from a given GenBank ID.

    Parameters:
    - genbank_id (str): The GenBank ID to fetch.

    Returns:
    - List[Tuple[str, int, int]]: A list of tuples where each tuple contains:
      * The product name or region type (str).
      * The start coordinate (int).
      * The end coordinate (int).
    """


    # Fetch the GenBank record
    handle = Entrez.efetch(db="nucleotide", id=genbank_id, rettype="gb", retmode="text")

    regions = []

    for record in SeqIO.parse(handle, "genbank"):
        for feature in record.features:

            # Extract CDS coordinates and product names
            if feature.type == "CDS":
                product = feature.qualifiers.get('product', ['unknown'])[0]
                if product not in ["ORF1ab polyprotein", "ORF1a polyprotein"]:
                    start = int(feature.location.start) + 1
                    end = int(feature.location.end)
                    regions.append((product, start, end))

            # Extract mat_peptide coordinates and product names
            elif feature.type == "mat_peptide":
                start = int(feature.location.start) + 1
                end = int(feature.location.end)
                product = feature.qualifiers.get('product', ['unknown'])[0]
                regions.append((product, start, end))

            # Extract UTR coordinates
            elif feature.type == "5'UTR" or feature.type == "3'UTR":
                start = int(feature.location.start) + 1
                end = int(feature.location.end)
                regions.append((feature.type, start, end))

    handle.close()

    # Remove duplicates and sort regions based on start coordinate
    regions = list(set(regions))
    regions.sort(key=lambda x: x[1])

    return regions

def plot_zna_regions(virus_name: str, sorted_keys: List[str],
                     regions_1: Dict[str, List[Tuple[int, int]]],
                     regions_2: Dict[str, List[Tuple[int, int]]],
                     genomic_regions_data: List[Tuple[str, int, int]],
                     colors: List[str],
                     figsize: Tuple[int, int] = (30, 15),
                     save: bool = False) -> None:
    """
    Plot Z-DNA regions predicted by two different methods, Z-HUNT and Z-DNABERT,
    across different strains of a virus species.

    Args:
    - virus_name (str): The name of the virus.
    - sorted_keys (list): A list of keys (strains names) used to order the data.
    - regions_1 (dict): A dictionary containing the Z-DNA regions predicted by Z-HUNT.
                      Each key is a strain name and each value is a list of start
                      and end positions of the Z-DNA regions.
    - regions_2 (dict): A dictionary containing the Z-DNA regions predicted by Z-DNABERT.
                      Structured the same as regions_1.
    - genomic_regions_data (list): A list of tuples where each tuple contains the name of a gene and its
                 start and end positions.
    - colors (list): A list of colors to use for the horizontal lines representing genes.
    - figsize (tuple, optional): The size of the figure to plot. Defaults to (30, 15).
    - save (bool, optional): If True, save the plot in PNG and PDF formats. Defaults to False.

    Returns:
    None. Shows the plot inline or saves it to file.
    """

    fig, ax = plt.subplots(figsize=figsize)
    genes, starts, ends = zip(*genomic_regions_data)

    for i in range(len(genomic_regions_data)):
        ax.hlines(y=0, xmin=starts[i], xmax=ends[i], linewidth=10, color=colors[i])

    lim = max(ends) + 50
    ax.set_xlim(0, lim)

    ax.yaxis.set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_position(('outward', 10))

    xticks = []
    xlabels = []
    for i in range(len(genomic_regions_data)):
        xticks.append((starts[i] + ends[i]) / 2)
        xlabels.append(genomic_regions_data[i][0])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=10)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    patches = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(genomic_regions_data))]
    ax.legend(patches, genes, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5, fontsize=10)

    plt.subplots_adjust(bottom=0.35)

    counter = 1
    for key in sorted_keys:
        if key in regions_1.keys():
            adata = [(regions_1[key][1:][i][0], regions_1[key][1:][i][1]) for i in range(len(regions_1[key][1:]))]
            if adata:
              starts, ends = zip(*adata)
              for i in range(len(adata)):
                  ax.hlines(y=counter, xmin=starts[i], xmax=ends[i], linewidth=5, colors='#e8071a', label='Z-HUNT')
              counter += 0.25

    counter = 1
    for key in sorted_keys:
        if key in regions_2.keys():
            adata = [(regions_2[key][1:][i][0], regions_2[key][1:][i][1]) for i in range(len(regions_2[key][1:]))]
            if adata:
              starts, ends = zip(*adata)
              for i in range(len(adata)):
                  ax.hlines(y=counter, xmin=starts[i], xmax=ends[i], linewidth=5, colors='#07a1e8', label='Z-DNABERT')
              counter += 0.25

    n_strains = len(regions_1)
    title = f"{virus_name}\n\n(number of strains: {n_strains})"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Position (nucleotides)', fontsize=12)

    if save:
        fig.savefig(f"{virus_name}.png", bbox_inches='tight')
        fig.savefig(f"{virus_name}.pdf", bbox_inches='tight')
    else:
        plt.show()

def extract_genbank_data(file_list: list) -> pd.DataFrame:
    """
    Extracts various data and calculations from a list of GenBank files.

    Parameters:
    - file_list (list): A list of paths to GenBank files.

    Returns:
    - pd.DataFrame: A DataFrame containing the extracted details for each GenBank file.
    """

    data = []

    for file_path in file_list:
        with open(file_path, "r") as handle:
            for record in SeqIO.parse(handle, "genbank"):
                # General Details
                accession_with_version = record.id
                virus_species_from_file = file_path.split("/")[-1].split(".")[0]

                # Taxonomy Details
                taxonomy = record.annotations.get('taxonomy', [])
                subfamily = taxonomy[8] if len(taxonomy) >= 9 else ""
                genus = taxonomy[9] if len(taxonomy) >= 10 else ""
                subgenus = taxonomy[10] if len(taxonomy) >= 11 else ""
                species = taxonomy[11] if len(taxonomy) >= 12 else ""

                # Source Features
                host = None
                collection_date = None
                for feature in record.features:
                    if feature.type == "source":
                        host = feature.qualifiers.get("host", [None])[0]
                        collection_date = feature.qualifiers.get("collection_date", [None])[0]
                        break
                if not collection_date:
                    collection_date = record.annotations.get('date', '')

                # Sequence Calculations
                seq_length = len(record.seq)
                gc_content = gc_fraction(record.seq) * 100
                melting_temperature = mt.Tm_NN(record.seq)

                gc_skew_values = GC_skew(record.seq, window=100)
                gc_skew_avg = sum(gc_skew_values) / len(gc_skew_values) if gc_skew_values else 0

                data.append([
                    virus_species_from_file, accession_with_version, subfamily,
                    genus, subgenus, species, host, seq_length, gc_content,
                    melting_temperature, gc_skew_avg,
                    collection_date
                ])

    columns = [
        'virus_species_from_file', 'accession_with_version', 'subfamily',
        'genus', 'subgenus', 'species', 'host', 'Sequence length', 'GC content',
        'Melting temperature', 'GC skew', 'date'
    ]
    df = pd.DataFrame(data, columns=columns)

    df['datetime'] = pd.to_datetime(df['date'], errors='coerce')

    return df

def plot_heatmap_with_dendrogram(clustered_dataframe: pd.DataFrame,
                                 row_linkage_matrix: np.ndarray,
                                 color_threshold: float,
                                 title: str = 'Heat map with dendrogram',
                                 figsize: Tuple[int, int] = (10, 10),
                                 save_figure: bool = False,
                                 file_name: str = 'results',
                                 close_figure: bool = False) -> None:

    # Validate input data
    if not isinstance(clustered_dataframe, pd.DataFrame) or clustered_dataframe.ndim != 2:
        raise ValueError("Input `clustered_dataframe` should be a 2D DataFrame.")

    if not is_valid_linkage(row_linkage_matrix):
        raise ValueError("Input `row_linkage_matrix` is not a valid linkage matrix.")

    if not np.issubdtype(clustered_dataframe.values.dtype, np.number):
        raise ValueError("Heatmap can only be plotted with numerical data.")

    # Create a figure to contain the plot elements
    figure = plt.figure(figsize=figsize)

    # Create a gridspec to handle the layout
    grid_spec = figure.add_gridspec(2, 2, width_ratios=[0.05, 1], height_ratios=[0.2, 1], wspace=0.02, hspace=0.02)

    # Add dendrogram on top
    dendrogram_axis = figure.add_subplot(grid_spec[0, 1])
    with plt.rc_context({'lines.linewidth': 0.5}):
        dendro = dendrogram(row_linkage_matrix, ax=dendrogram_axis, orientation='top', color_threshold=color_threshold)
    dendrogram_axis.axis('off')

    # Create a color map
    clusters = fcluster(row_linkage_matrix, color_threshold, criterion='distance')
    unique_clusters = len(np.unique(clusters))

    colors = sns.color_palette('tab20', n_colors=unique_clusters)
    color_map = dict(enumerate(colors, 1))

    # Change the color of each line to match the cluster colors
    for i, d, c in zip(dendro['icoord'], dendro['dcoord'], clusters):
        for j in range(4):
            x = 0.5 * sum(i[j:j+2])
            y = d[j]
            dendrogram_axis.plot(x, y, color=color_map[c])

    # Add heatmap
    heatmap_axis = figure.add_subplot(grid_spec[1, 1])
    sns.heatmap(clustered_dataframe, annot=False, ax=heatmap_axis, cbar=False, xticklabels=False, yticklabels=False)

    # Add title to the entire figure
    figure.suptitle(title, fontsize=10, y=0.91)

    plt.tick_params(labelsize=5)
    if save_figure:
        plt.savefig(f"{file_name}_heatmap.png")
        plt.savefig(f"{file_name}_heatmap.pdf")
    if close_figure:
        plt.close()
    plt.show()

def optimal_dendrogram_threshold_and_clusters(linkage_matrix: np.ndarray, N: int) -> float:
    """
    Find the threshold for the linkage matrix such that the number of clusters is
    approximately N. If a suitable threshold cannot be found, return a default value.
    """

    # Starting threshold
    threshold = linkage_matrix[-N, 2] if N < linkage_matrix.shape[0] else linkage_matrix[0, 2]

    # Small value to adjust the threshold in each step
    delta = 0.001

    # Default threshold as the average of all distances
    default_threshold = np.mean(linkage_matrix[:, 2])

    while True:
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        unique_clusters = len(np.unique(clusters))

        if unique_clusters == N:
            return threshold

        if unique_clusters < N:
            threshold -= delta
        else:
            threshold += delta

        # Return default threshold if search goes out of bounds
        if threshold < 0 or threshold > linkage_matrix[-1, 2]:
            print("Warning: Couldn't find a suitable threshold for the specified number of clusters.")
            print(f"Using a default threshold of {default_threshold}")
            return default_threshold

def plot_time_length_regression(
    virus_name: str,
    meta_df: pd.DataFrame,
    intervals_column: str,
    color_threshold: float = None,
    clustered_dataframe: pd.DataFrame = None,
    row_linkage_matrix: np.ndarray = None,
    grouping_column: Optional[str] = None,
    title: str = 'Time and ZNA length regression',
    legend: str = 'Cluster: ',
    remove_outliers: Tuple[bool, int] = (False, 3),
    figsize: Tuple[int, int] = (10, 5),
    point_size: int = 15,
    save_figure: bool = False,
    normalize: bool = False,
    file_name: str = 'results',
    legend_loc: str = 'best',
    close_figure: bool = False) -> None:
    """
    Plot a time-length regression of Z-RNA regions' length against the collection date for different lineages.

    Parameters:
    - virus_name (str): Name of the virus to filter data.
    - meta_df (pd.DataFrame): DataFrame containing the data to be plotted.
    - intervals_column (str): The column name for length intervals in `meta_df`.
    - clustered_dataframe (pd.DataFrame): DataFrame with clustered data.
    - row_linkage_matrix (np.ndarray): Linkage matrix for hierarchical clustering.
    - color_threshold (float): Threshold to use for coloring clusters.
    - grouping_column (Optional[str]): The column for grouping data; defaults to None.
    - title (str): Title of the plot; defaults to 'Time and ZNA length regression'.
    - legend (str): Base string for the legend of each cluster; defaults to 'Cluster: '.
    - remove_outliers (Tuple[bool, int]): Indicator and z-score threshold for outlier removal; defaults to (False, 3).
    - figsize (Tuple[int, int]): Figure size; defaults to (10, 5).
    - point_size (int): The size of points in the scatter plot; defaults to 8.
    - save_figure (bool): Whether to save the figure as a PNG; defaults to False.
    - normalize (bool): Whether to normalize the data; defaults to False.
    - file_name (str): The name of the file to save the figure; defaults to 'results'.
    - legend_loc (str): The location of the legend; defaults to 'best'.
    - close_figure (bool): Whether to close the figure after plotting; defaults to False.

    Returns:
    None

    This function first processes the input data to ensure it's ready for plotting.
    It then uses this data to plot the time-length regression and optionally a regression line.
    The function can handle and visualize different lineage clusters,
    providing a comprehensive view of the data spread across time and length.
    Additional features include outlier removal, data normalization, and customized figure saving.
    """

    # Make an explicit copy
    df_copy = meta_df[meta_df['virus_species_from_file'] == virus_name].copy()

    # Convert 'date' to datetime format only once
    df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')

    # Normalize the data if specified
    if normalize:
        df_copy['Normalized_Length'] = df_copy[intervals_column] / df_copy['Sequence Length']
        length_column = 'Normalized_Length'
    else:
        length_column = intervals_column

    # Convert the collection date to datetime and ordinal
    df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['date'])
    df_copy['Date_Ordinal'] = df_copy['date'].apply(lambda x: x.toordinal())

    # Remove outliers if specified
    if remove_outliers[0]:
        z_scores = df_copy[['Date_Ordinal', intervals_column]].apply(
            lambda x: (x - x.mean()) / x.std()
        )
        df_copy = df_copy[
            (np.abs(z_scores['Date_Ordinal']) <= remove_outliers[1]) &
            (np.abs(z_scores[intervals_column]) <= remove_outliers[1])
        ]

    # Fit a linear regression model
    X = df_copy[['Date_Ordinal']]
    y = df_copy[length_column]
    regression_model = LinearRegression()
    regression_model.fit(X, y)
    y_pred = regression_model.predict(X)

    # Create the figure
    plt.figure(figsize=figsize)

    if grouping_column != None:
        unique_lineages = df_copy[grouping_column].unique()
        colors = mpl.rcParams['axes.prop_cycle'].by_key()['color'][1:len(unique_lineages)+2]
        color_map = dict(zip(sorted(unique_lineages), colors))
        # Plot the data points for each cluster
        for linage in set(df_copy[grouping_column]):
            plt.scatter(df_copy[df_copy[grouping_column] == linage]['date'],
                        df_copy[df_copy[grouping_column] == linage][length_column],
                        color=color_map[linage], label=f"{linage}", s=point_size)
    else:
        grouping_column = 'leaves_color'
        dendro = dendrogram(row_linkage_matrix, color_threshold=color_threshold, no_plot=True)
        d = dict(zip(clustered_dataframe.columns, dendro['leaves_color_list']))
        df_copy['leaves_color'] = df_copy['accession_with_version'].apply(lambda x: d.get(x))

        unique_lineages = df_copy[grouping_column].unique()
        colors = mpl.rcParams['axes.prop_cycle'].by_key()['color'][1:len(unique_lineages)+2]
        color_map = dict(zip(sorted(unique_lineages), colors))
        # Plot the data points for each cluster
        for linage in set(df_copy[grouping_column]):
            plt.scatter(df_copy[df_copy[grouping_column] == linage]['date'],
                        df_copy[df_copy[grouping_column] == linage][length_column],
                        color=color_map[linage], label=f"{linage}", s=point_size)

    # Plot the regression line
    plt.plot(df_copy['date'], y_pred, color='red', label='Regression Line')

    # Get the slope and intercept of the regression line
    slope = regression_model.coef_[0]
    intercept = regression_model.intercept_

    # Create a list of legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f"{legend} {linage}",
               markerfacecolor=color, markersize=10)
        for linage, color in color_map.items()
    ]
    legend_elements.append(Line2D([0], [0], color='red', lw=2, label='Regression Line'))
    legend_elements.append(Line2D([0], [0], marker='None', color='w',
                                  label=f"Slope = {slope:.2f}\nIntercept = {intercept:.2f}"))

    # Customize the plot
    plt.legend(handles=legend_elements, loc=legend_loc, framealpha=0.5)
    plt.grid(visible=True, which='major', axis='both', linestyle='-')
    plt.title(title)
    plt.xlabel('Collection Date')
    plt.ylabel('Z-RNA regions total length')

    # Save the figure if specified
    if save_figure:
        plt.savefig(f"{file_name}_heatmap.png")
        plt.savefig(f"{file_name}_heatmap.pdf")

    # Close the figure if specified
    if close_figure:
        plt.close()

    # Show the plot
    plt.show()

def remove_outliers(df: pd.DataFrame, column_names: list, multiplier: float = 3) -> pd.DataFrame:
    """
    Removes outliers from a DataFrame based on values in specified columns using the IQR method.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        column_names (list): A list of column names in which to check for outliers.
        multiplier (float): The multiplier for the IQR. Defaults to 1.5.
        Increase this value to be more lenient with outliers, and decrease it to be stricter.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """

    for column_name in column_names:
        # Calculate Q1, Q2, and IQR
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Filter the data frame
        df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return df

def draw_boxplot_species(
        data_subset: pd.DataFrame,
        category_label: str = 'Virus name',
        value_label: str = 'ZDNABERT intervals total length',
        plot_title: str = '',
        figure_size: Tuple[int, int] = (15, 6),
        text_position: Tuple[float, float] = (0.02, 0.02),
        legend_location: str = 'best') -> None:
    """
    Draws a boxplot for certain categories.

    Parameters:
    data_subset (pd.DataFrame): The subset of the data to be plotted.
    category_label (str): The name of the column representing the category.
    value_label (str): The name of the column representing the values.
    plot_title (str): The title of the plot.
    figure_size (Tuple[int, int]): The size of the figure.
    text_position (Tuple[float, float]): The position of the text inside the plot.
    legend_location (str): The location of the legend.

    Returns:
    None
    """
    # The unique categories
    categories = list(data_subset[category_label].unique())

    # Mapping English letters to Greek letters
    greek_dict = {
        'alphacoronavirus': r'$\alpha$',
        'betacoronavirus': r'$\beta$',
        'gammacoronavirus': r'$\gamma$',
        'deltacoronavirus': r'$\delta$',
        'alphapironavirus': r'$\pi$',
    }

    # Specify the species groups
    species_groups = ['Bats', 'Birds', 'Cats', 'Chicken', 'Hedgehog', 'Human',
                      'Human/Swine/Dog', 'Mice', 'Mink', 'Myodes', 'Rat', 'Salmon',
                      'Shrew', 'Swine', 'Whale']

    # Filter the DataFrame to only include rows where 'species' is in species_groups
    data_subset = data_subset[data_subset['Host'].isin(species_groups)]

    # Sort dataframe by species
    data_subset = data_subset.sort_values(by=['Host', 'virus_species_from_file'])

    # Store the 'Intervals total length' for each category
    category_values = []
    for category in categories:
        values = data_subset[data_subset[category_label] == category][value_label].values
        if not np.isnan(values).any():
            category_values.append(values)

    # Assuming that this function exists
    category_values = filter_arrays(category_values)

    # Perform the ANOVA
    f_value, p_value = stats.f_oneway(*category_values)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=figure_size)

    # Create the boxplot with seaborn
    box_plot = sns.boxplot(
        x=category_label,
        y=value_label,
        hue='Host',
        data=data_subset,
        dodge=False
    )

    # Add a title and labels
    ax.set_title(plot_title)
    ax.set_xlabel('Viruses')
    ax.set_ylabel(f'Z-RNA {value_label}')

    # Get color palette
    palette = sns.color_palette("husl", 20)

    # Create a mapping of labels to species
    label_to_species = data_subset.groupby(category_label)['Host'].agg(pd.Series.mode).to_dict()

    # Create a mapping of species to colors
    species_to_colors = dict(zip(species_groups, palette))

    # Mapping taxa to colors
    taxa_to_colors = {
        'alphapironavirus': '#FF6B6B',  # Light Red
        'alphacoronavirus': '#4ECDC4',  # Turquoise
        'betacoronavirus': '#556270',   # Dark Grayish Blue
        'deltacoronavirus': '#C7F464',  # Light Yellow Green
        'gammacoronavirus': '#FFA577'   # Light Orange
    }

    # Apply the colors to the labels based on the most common species for that label
    for i, label_text in enumerate(ax.get_xticklabels()):
        species = label_to_species[label_text.get_text()]
        color = species_to_colors[species]
        label_text.set_color(color)

        # Extract the corresponding 'taxa' value for the label
        taxa = data_subset[data_subset[category_label] == label_text.get_text()]['genus'].unique()[0]
        # Convert first letter to Greek equivalent
        taxa = taxa.lower()
        taxa_greek = greek_dict.get(taxa, taxa[0])

        # Update y position to below y=0 and set color based on taxa
        ax.text(
            (i + 0.5) / len(categories), 0.01, taxa_greek,
            horizontalalignment='center', size='small', color='black',
            weight='semibold', transform=ax.transAxes,
            bbox=dict(facecolor=taxa_to_colors[taxa], alpha=0.4, boxstyle='round,pad=0.2')
        )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10, fontweight='bold')

    # Set the legend title
    ax.legend(title='Host species', loc=legend_location)

    # Add the F-value, p-value, and test name inside the plot
    anova_text = f'ANOVA Test\nF-value: {f_value:.2f}\nP-value: {p_value:.4f}'
    ax.text(
        *text_position, anova_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )

    # Show the plot
    plt.show()

def pango_to_who(pango_lineage: str) -> str:
    """
    Converts Pango lineage of SARS-CoV-2 virus to WHO label.

    Args:
        pango_lineage (str): A string representing the Pango lineage of the virus.

    Returns:
        str: A string representing the WHO label.
    """
    pango_lineage = pango_lineage.upper()

    pango_to_who_map = {
        "B.1.1.7": "Alpha",
        "Q": "Alpha",
        "B.1.351": "Beta",
        "P.1": "Gamma",
        "B.1.617.2": "Delta",
        "B.1.617.1": "Kappa",
        "B.1.617.3": "Kappa",
        "B.1.427": "Epsilon",
        "B.1.429": "Epsilon",
        "B.1.525": "Eta",
        "B.1.526": "Iota",
        "C.37": "Lambda",
        "B.1.621": "Mu",
        "B.1.621.1": "Mu",
        "BA.1": "Omicron",
        "BA.2": "Omicron",
        "BA.4": "Omicron",
        "BA.5": "Omicron",
        "BA.2.12.1": "Omicron",
        "BA.2.75": "Omicron",
        "BQ.1": "Omicron",
        "XBB.1.5": "Omicron",
        "XBB.1.16": "Omicron",
        "P.2": "Zeta"
    }

    for pango, who in pango_to_who_map.items():
        if pango_lineage.startswith(pango):
            return who

    return "Unknown"

def draw_boxplot_who(
        df_subset: pd.DataFrame,
        label: str = 'WHO',
        values: str = 'Intervals Total Length',
        title: str = '',
        figsize: Tuple[int, int] = (15, 6),
        pos: Tuple[float, float] = (0.02, 0.02)) -> None:
    """
    Draws a boxplot for a given DataFrame.

    Parameters:
    df_subset (pd.DataFrame): The subset of the data to be plotted.
    label (str): The name of the column representing the category.
    values (str): The name of the column representing the values.
    title (str): The title of the plot.
    figsize (Tuple[int, int]): The size of the figure.
    pos (Tuple[float, float]): The position of the text inside the plot.

    Returns:
    None
    """
    # Pangolin values
    pangolin_values = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon',
                       'Eta', 'Iota', 'Kappa', 'Lambda', 'Mu',
                       'Omicron', 'Unknown', 'Zeta']

    # Create an empty list to store the 'Intervals total length' for each 'Pangolin' value
    data = []

    # For each 'Pangolin' value, append the 'Intervals total length' to the data list
    for pangolin in pangolin_values:
        data.append(df_subset[df_subset[label] == pangolin][values].values)

    # Assuming that this function exists
    data = filter_arrays(data)

    # Perform the ANOVA
    f_value, p_value = stats.f_oneway(*data)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create the boxplot with seaborn
    sns.boxplot(x=label, y=values, data=df_subset, order=pangolin_values, ax=ax)

    # Add a title and labels
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel(f'Z-RNA {values}')

    # Add the F-value, p-value, and test name inside the plot
    anova_text = f'ANOVA Test\nF-value: {f_value:.2f}\nP-value: {p_value:.2f}'
    ax.text(
        *pos, anova_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )

    # Show the plot
    plt.show()

def plot_time_length_linages_boxplot(
    meta_df: pd.DataFrame,
    length_column: str = '',
    title: str = '',
    remove_outliers: Tuple[bool, int] = (False, 3),
    figsize: Tuple[int, int] = (14, 8),
    text_position: Tuple[float, float] = (0.01, 0.90),
    save_figure: bool = False,
    normalize: bool = False,
    file_name: str = 'results',
    legend_loc: str = 'best',
    close_figure: bool = False
) -> None:

    # Copy the dataframe
    df_copy = meta_df.copy()

    # Normalize the data if specified
    if normalize:
        df_copy['Normalized_Length'] = df_copy['Intervals Total Length'] / df_copy['Sequence Length']
        length_column = 'Normalized_Length'

    # Convert the collection date to datetime and extract year and month
    df_copy['datetime'] = pd.to_datetime(df_copy['datetime'], errors='coerce')
    df_copy['Year_Month'] = df_copy['datetime'].dt.to_period('M')

    # Order by year and month
    df_copy['Year_Month'] = df_copy['Year_Month'].astype(
        CategoricalDtype(
            categories=sorted(df_copy['Year_Month'].unique()),
            ordered=True
        )
    )

    # Remove outliers if specified
    if remove_outliers[0]:
        z_scores = df_copy[[length_column]].apply(lambda x: (x - x.mean()) / x.std())
        df_copy = df_copy[(np.abs(z_scores[length_column]) <= remove_outliers[1])]

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the boxplot for each month using seaborn
    sns.boxplot(x="Year_Month", y=length_column, data=df_copy, palette='viridis', width=0.5, ax=ax)

    # Perform ANOVA
    groups = [group[length_column].dropna() for name, group in df_copy.groupby('Year_Month')]
    f_value, p_value = stats.f_oneway(*groups)

    # Add the F-value, p-value, and test name inside the plot
    anova_text = f'ANOVA Test\nF-value: {f_value:.2f}\nP-value: {p_value:.4f}'
    ax.text(
        *text_position, anova_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )

    # Customize the plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Collection Date', fontsize=14)
    ax.set_ylabel('Z-RNA Regions Length', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)

    # Save the figure if specified
    if save_figure:
        plt.savefig(f"{file_name}_boxplot.png", dpi=300)
        plt.savefig(f"{file_name}_boxplot.pdf", dpi=300)

    # Close the figure if specified
    if close_figure:
        plt.close()

    # Show the plot
    plt.show()

def plot_time_length_regression_linages(
    meta_df: pd.DataFrame,
    length_column: str = '',
    category_column: str = 'WHO',
    title: str = '',
    remove_outliers: Tuple[bool, int] = (False, 3),
    figsize: Tuple[int, int] = (10, 5),
    point_size: int = 3,
    save_figure: bool = False,
    normalize: bool = False,
    file_name: str = 'results',
    legend_loc: str = 'best',
    close_figure: bool = False,
    show_legend: bool = True
) -> None:
    """
    Plot a time-length regression of Z-RNA regions' length against the collection date for different lineages.

    Parameters:
    meta_df (pd.DataFrame): The dataframe containing the data to be plotted.
    title (str, optional): The title of the plot. Defaults to 'Time and ZNA length regression'.
    remove_outliers (Tuple[bool, int], optional): A tuple indicating whether to remove outliers
    and the z-score threshold for outlier removal. Defaults to (False, 3).
    figsize (Tuple[int, int], optional): The size of the figure. Defaults to (10, 5).
    point_size (int, optional): The size of the points in the scatter plot. Defaults to 3.
    save_figure (bool, optional): Whether to save the figure as a PNG. Defaults to False.
    normalize (bool, optional): Whether to normalize the data. Defaults to False.
    file_name (str, optional): The name of the file to save the figure as. Defaults to 'results'.
    legend_loc (str, optional): The location of the legend. Defaults to 'best'.
    close_figure (bool, optional): Whether to close the figure after plotting. Defaults to False.

    Returns:
    None
    """

    # Copy the dataframe
    df_copy = meta_df.copy()

    # Normalize the data if specified
    if normalize:
        df_copy['Normalized_Length'] = df_copy['Intervals Total Length'] / df_copy['Sequence Length']
        length_column = 'Normalized_Length'

    # Convert the collection date to datetime and ordinal
    df_copy['Collection_Date'] = pd.to_datetime(df_copy['datetime'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Collection_Date'])
    df_copy['Date_Ordinal'] = df_copy['Collection_Date'].apply(lambda x: x.toordinal())

    # Remove outliers if specified
    if remove_outliers[0]:
        z_scores = df_copy[['Date_Ordinal', 'Intervals Total Length']].apply(
            lambda x: (x - x.mean()) / x.std()
        )
        df_copy = df_copy[
            (np.abs(z_scores['Date_Ordinal']) <= remove_outliers[1]) &
            (np.abs(z_scores['Intervals Total Length']) <= remove_outliers[1])
        ]

    # Create a color map
    unique_values = df_copy[category_column].unique()
    colors = get_high_contrast_colors_gr(len(unique_values))
    color_map = dict(zip(unique_values, colors))

    # Fit a linear regression model
    X = df_copy[['Date_Ordinal']]
    y = df_copy[length_column]
    regression_model = LinearRegression()
    regression_model.fit(X, y)
    y_pred = regression_model.predict(X)

    # Create the figure
    plt.figure(figsize=figsize)

    # Plot the data points for each cluster
    for linage in set(df_copy[category_column]):
        plt.scatter(df_copy[df_copy[category_column] == linage]['Collection_Date'],
                    df_copy[df_copy[category_column] == linage][length_column],
                    color=color_map[linage], label=f"{linage}", s=point_size)

    # Plot the regression line
    plt.plot(df_copy['Collection_Date'], y_pred, color='red', label='Regression Line')

    # Get the slope and intercept of the regression line
    slope = regression_model.coef_[0]
    intercept = regression_model.intercept_

    # Create a list of legend elements
    if show_legend:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=f"{linage}",
                   markerfacecolor=color, markersize=10)
            for linage, color in color_map.items()
        ]
        legend_elements.append(Line2D([0], [0], color='red', lw=2, label='Regression Line'))
        legend_elements.append(Line2D([0], [0], marker='None', color='w',
                                      label=f"Slope = {slope:.2f}\nIntercept = {intercept:.2f}"))
        plt.legend(handles=legend_elements, loc=legend_loc, framealpha=0.5)

    # Customize the plot
    plt.grid(visible=True, which='major', axis='both', linestyle='-')
    plt.title(title)
    plt.xlabel('Collection Date')
    plt.ylabel('Z-RNA Regions Length')

    # Save the figure if specified
    if save_figure:
        plt.savefig(f"{file_name}_heatmap.png")
        plt.savefig(f"{file_name}_heatmap.pdf")

    # Close the figure if specified
    if close_figure:
        plt.close()

    # Show the plot
    plt.show()