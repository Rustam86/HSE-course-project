import os
import subprocess
import tempfile
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
from Bio import Entrez, SeqIO
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy import ndimage, stats
from sklearn.linear_model import LinearRegression
from torch import nn
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizer


def seq2kmer(seq: str, k: int) -> List[str]:
    """
    Converts a sequence into a list of k-mers.
    
    Parameters:
    seq (str): The sequence to convert.
    k (int): The length of the k-mers.
    
    Returns:
    List[str]: A list of k-mers.
    """
    return [seq[x:x+k] for x in range(len(seq)+1-k)]


def split_seq(seq: str, length: int = 512, pad: int = 16) -> List[str]:
    """
    Splits a sequence into smaller pieces.
    
    Parameters:
    seq (str): The sequence to split.
    length (int): The length of the pieces.
    pad (int): The overlap between the pieces.
    
    Returns:
    List[str]: A list of sequence pieces.
    """
    return [seq[st:min(st+512, len(seq))] for st in range(0, len(seq), length-pad)]


def stitch_np_seq(np_seqs: List[np.ndarray], pad: int = 16) -> np.ndarray:
    """
    Stitches together the predictions for each piece of the sequence.
    
    Parameters:
    np_seqs (List[np.ndarray]): A list of numpy arrays containing the predictions.
    pad (int): The overlap between the pieces.
    
    Returns:
    np.ndarray: A numpy array containing the stitched predictions.
    """
    res = np.array([])
    for seq in np_seqs:
        res = np.concatenate([res[:-pad], seq])
    return res


def run_zdnabert(sequence: str, model: BertForTokenClassification, tokenizer: BertTokenizer, model_confidence_threshold: float = 0.2, minimum_sequence_length: int = 10) -> dict:
    """
    Predicts Z-DNA regions in a given sequence using a trained model.

    Parameters:
        sequence (str): The DNA sequence to analyze.
        model (BertForTokenClassification): The trained model.
        tokenizer (BertTokenizer): The tokenizer.
        model_confidence_threshold (float): The threshold for the model's confidence.
        minimum_sequence_length (int): The minimum length of a sequence to consider.

    Returns:
        dict: A dictionary where the keys are the sequence names and the values are lists of tuples representing the predicted Z-DNA regions.
    """
    result_dict = {}

    # Convert sequence to k-mers
    kmer_seq = seq2kmer(sequence.upper(), 6)
    # Split the sequence into smaller pieces
    seq_pieces = split_seq(kmer_seq)

    with torch.no_grad():
        # Iterate over each sequence piece and perform Z-DNA prediction
        preds = [torch.softmax(
            model(
                torch.LongTensor(
                    tokenizer.encode(' '.join(seq_piece), add_special_tokens=False)).unsqueeze(0)
            ).squeeze()[:, 1].cpu().numpy()) for seq_piece in seq_pieces]
        result_dict['sequence'] = stitch_np_seq(preds)

    labeled_regions = {}
    for seq_name, predictions in result_dict.items():
        # Label connected regions above the confidence threshold
        labeled, max_label = scipy.ndimage.label(predictions > model_confidence_threshold)
        # Extract regions longer than the minimum sequence length
        labeled_regions[seq_name] = [(candidate[0], candidate[-1]) for label in range(1, max_label + 1) for candidate in [np.where(labeled == label)[0]] if
                                     candidate.shape[0] > minimum_sequence_length]

    return labeled_regions


def run_zhunt(sequence: str, zhunt_path: str, window_size: int = 6, min_size: int = 3, max_size: int = 6) -> List[Tuple[int, int]]:
    """
    Run the ZHunt program to predict Z-DNA forming regions in a DNA sequence.

    Parameters:
    sequence (str): The DNA sequence to analyze.
    zhunt_path (str): The path to the ZHunt executable.
    window_size (int): The window size for the ZHunt program. Default is 6.
    min_size (int): The minimum size for the ZHunt program. Default is 3.
    max_size (int): The maximum size for the ZHunt program. Default is 6.

    Returns:
    List[Tuple[int, int]]: A list of tuples representing the start and end positions of the predicted Z-DNA forming regions.
    """
    # Ensure the sequence only contains valid DNA bases
    assert set(sequence).issubset({"A", "C", "G", "T", "N"}), "Invalid DNA sequence"

    # Create a temporary file
    file_descriptor, temp_file_path = tempfile.mkstemp()
    os.close(file_descriptor)

    try:
        # Write the sequence to the temporary file
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(sequence)

        # Run the ZHunt program
        subprocess.run(
            [zhunt_path, str(window_size), str(min_size), str(max_size), temp_file_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            input=sequence, encoding='ascii'
        )

        # Read the ZHunt output into a DataFrame
        with open(temp_file_path + ".Z-SCORE", 'r') as zhunt_output:
            output_data = pd.read_csv(zhunt_output,
                             names=['Start', 'End', 'nu-1', 'nu-2', 'nu-3', 
                                    'ZH-Score', 'Sequence', 'Conformation'],
                             skiprows=1, sep='\s+')

        # Filter the DataFrame to only include rows with a ZH-Score greater than 500
        filtered_data = output_data[output_data['ZH-Score'] > 500]

        # Return a list of tuples representing the start and end positions of the predicted Z-DNA forming regions
        return list(zip(filtered_data['Start'], filtered_data['End']))

    except Exception as error:
        print(f"An error occurred while running ZHunt: {error}")

    finally:
        # Clean up the temporary files
        os.remove(temp_file_path)
        os.remove(temp_file_path + ".Z-SCORE")


def parse_prediction_files(file_path: str) -> dict:
    """
    Parse output files from Zdnabert and Zhunt.

    This function reads a file, parses the contents, and stores
    the information in a dictionary. The keys of the dictionary 
    are the lines from the file that do not start with a space, 
    and the values are lists of lists, each containing numbers 
    extracted from lines starting with three spaces.

    Parameters:
    file_path (str): The path to the file to be parsed.

    Returns:
    dict: A dictionary containing parsed data.
    """
    # Open the file
    with open(file_path, 'r') as file:
        data_dict = {}
        key = ''
        for line in file:
            # If line ends with '.fas', get the spec by stripping '.fas'
            if line.strip()[-4:] == '.fas':
                spec = line.strip()[:-4]
                continue

            # If line starts with "100%|", skip the line
            if line.startswith("100%|"):
                continue

            # If line doesn't start with '  ' or '   ', make it a key
            if not (line.startswith('  ') or line.startswith('   ')):
                key = line.strip()
                data_dict[key] = []
            # If line starts with '   ', convert it to a list of floats and add it to the current key
            elif line.startswith('   '):
                value = [float(i) for i in line.strip().split()]
                data_dict[key].append(value)

        # If a key has only one value, append a list [0, 0, False] to it
        for key in data_dict:
            if len(data_dict[key]) == 1:
                data_dict[key].append([0, 0, False])

        # Remove the key '' if it exists
        try:
            data_dict.pop('')
        except KeyError:
            pass
    
    return data_dict



def compute_jaccard_index(set_intervals1: List[Tuple[int, int]], set_intervals2: List[Tuple[int, int]]) -> float:
    """
    This function computes the Jaccard index between two sets of intervals.

    Parameters:
    set_intervals1, set_intervals2 (list of tuples): The intervals to be compared.

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



def create_clustered_dataframe(zrna_intervals_file_path: str, virus_name: str) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    This function creates a clustered dataframe based on Jaccard indices of intervals.

    Parameters:
    zrna_intervals_file_path (str): The file containing Z-RNA interval data.
    virus_name (str): The name of the virus.
    dataframe_subset (pd.DataFrame): The subset of the dataframe to be clustered.

    Returns:
    clustered_dataframe (pd.DataFrame): The clustered dataframe.
    row_linkage_matrix (np.ndarray): The hierarchical clustering encoded as a linkage matrix.
    optimal_color_threshold (int): The optimal color threshold for the clusters.
    """

    # Parse the intervals file
    intervals_dict = parse_prediction_files(zrna_intervals_file_path)
    sequence_ids = list(intervals_dict.keys())

    # Check if there are enough sequences
    if len(sequence_ids) < 5:
        return 'Less than 5 sequences'

    # Compute the Jaccard index matrix
    intervals_list = list(intervals_dict.items())
    interval_count = len(intervals_list)
    sequence_labels = [intervals_list[i][0] for i in range(interval_count)]
    jaccard_index_matrix = [[compute_jaccard_index(intervals_list[i][1], intervals_list[j][1]) for j in range(interval_count)] for i in range(interval_count)]
    jaccard_dataframe = pd.DataFrame(jaccard_index_matrix, index=sequence_labels, columns=sequence_labels)

    # Perform hierarchical clustering on rows and columns
    row_linkage_matrix = linkage(jaccard_dataframe.values, method='average', metric='euclidean')
    column_linkage_matrix = linkage(jaccard_dataframe.values.T, method='average', metric='euclidean')

    # Reorder the dataframe based on the clustering
    row_dendrogram = dendrogram(row_linkage_matrix, no_plot=True)
    column_dendrogram = dendrogram(column_linkage_matrix, no_plot=True)
    clustered_dataframe = jaccard_dataframe.iloc[row_dendrogram['leaves'], column_dendrogram['leaves']]

    # Find the optimal color threshold for the clusters
    max_cluster_count = -1
    optimal_color_threshold = -1
    for color_threshold in range(2, 7):
        cluster_index = fcluster(row_linkage_matrix, t=color_threshold, criterion='distance')
        cluster_count = len(set(cluster_index))
        if cluster_count <= 10 and cluster_count > max_cluster_count:
            max_cluster_count = cluster_count
            optimal_color_threshold = color_threshold

    return clustered_dataframe, row_linkage_matrix, optimal_color_threshold



def create_clustered_dataframe(zrna_intervals_file_path: str, virus_name: str) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    This function creates a clustered dataframe based on Jaccard indices of intervals.

    Parameters:
    zrna_intervals_file_path (str): The file containing Z-RNA interval data.
    virus_name (str): The name of the virus.
    dataframe_subset (pd.DataFrame): The subset of the dataframe to be clustered.

    Returns:
    clustered_dataframe (pd.DataFrame): The clustered dataframe.
    row_linkage_matrix (np.ndarray): The hierarchical clustering encoded as a linkage matrix.
    optimal_color_threshold (int): The optimal color threshold for the clusters.
    """

    # Parse the intervals file
    intervals_dict = parse_prediction_files(zrna_intervals_file_path)
    sequence_ids = list(intervals_dict.keys())

    # Check if there are enough sequences
    if len(sequence_ids) < 5:
        return 'Less than 5 sequences'

    # Compute the Jaccard index matrix
    intervals_list = list(intervals_dict.items())
    interval_count = len(intervals_list)
    sequence_labels = [intervals_list[i][0] for i in range(interval_count)]
    jaccard_index_matrix = [[compute_jaccard_index(intervals_list[i][1], intervals_list[j][1]) for j in range(interval_count)] for i in range(interval_count)]
    jaccard_dataframe = pd.DataFrame(jaccard_index_matrix, index=sequence_labels, columns=sequence_labels)

    # Perform hierarchical clustering on rows and columns
    row_linkage_matrix = linkage(jaccard_dataframe.values, method='average', metric='euclidean')
    column_linkage_matrix = linkage(jaccard_dataframe.values.T, method='average', metric='euclidean')

    # Reorder the dataframe based on the clustering
    row_dendrogram = dendrogram(row_linkage_matrix, no_plot=True)
    column_dendrogram = dendrogram(column_linkage_matrix, no_plot=True)
    clustered_dataframe = jaccard_dataframe.iloc[row_dendrogram['leaves'], column_dendrogram['leaves']]

    # Find the optimal color threshold for the clusters
    max_cluster_count = -1
    optimal_color_threshold = -1
    for color_threshold in range(2, 7):
        cluster_index = fcluster(row_linkage_matrix, t=color_threshold, criterion='distance')
        cluster_count = len(set(cluster_index))
        if cluster_count <= 10 and cluster_count > max_cluster_count:
            max_cluster_count = cluster_count
            optimal_color_threshold = color_threshold

    return clustered_dataframe, row_linkage_matrix, optimal_color_threshold




def plot_time_length_regression(meta_df: pd.DataFrame, clusterd_df: pd.DataFrame, row_linkage: np.ndarray, color_threshold: float,
                                title: str = 'Time and ZNA length regression',
                                remove_outliers: Tuple[bool, int] = (False, 3),
                                figsize: Tuple[int, int] = (10, 5), point_size: int = 3,
                                save_figure: bool = False, normalize: bool = False, file_name: str = 'results',
                                legend_loc: str = 'best', close_figure: bool = False) -> None:
    """
    This function plots a time-length regression.

    Parameters:
    df (pd.DataFrame): The dataframe to be plotted.
    row_linkage (np.ndarray): The hierarchical clustering encoded as a linkage matrix.
    color_threshold (float): The color threshold for the dendrogram.
    title (str): The title of the plot.
    remove_outliers (Tuple[bool, int]): A tuple indicating whether to remove outliers and the z-score threshold for outlier removal.
    figsize (Tuple[int, int]): The size of the figure.
    point_size (int): The size of the points in the scatter plot.
    save_figure (bool): Whether to save the figure as a PNG.
    normalize (bool): Whether to normalize the data.
    file_name (str): The name of the file to save the figure as.
    legend_loc (str): The location of the legend.
    off (bool): Whether to close the figure after plotting.

    Returns:
    None
    """

    # Copy the dataframe
    clusterd_ids = clusterd_df.columns
    clusterd_ids_dict = {k: v for v, k in enumerate(clusterd_ids)}
    df_copy = meta_df[meta_df['Accession'].isin(clusterd_ids)]
    df_copy['order'] = df_copy['Accession'].apply(clusterd_ids_dict.get)
    df_copy = df_copy.sort_values('order').drop(columns='order')


    # Normalize the data if specified
    if normalize:
        df_copy['Normalized_Length'] = df_copy['Intervals Total Length'] / df_copy.loc[df_copy['Accession'].isin(df_copy['Accession']), 'Sequence Length']
        length_column = 'Normalized_Length'
    else:
        length_column = 'Intervals Total Length'

    # Assign each data point to a cluster
    df_copy['Clusters'] = fcluster(row_linkage, t=color_threshold, criterion='distance')

    # Convert the collection date to datetime and ordinal
    df_copy['Collection_Date'] = pd.to_datetime(df_copy['Collection Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Collection_Date'])
    df_copy['Date_Ordinal'] = df_copy['Collection_Date'].apply(lambda x: x.toordinal())

    # Remove outliers if specified
    if remove_outliers[0]:
        z_scores = df_copy[['Date_Ordinal', 'Intervals Total Length']].apply(lambda x: (x - x.mean()) / x.std())
        df_copy = df_copy[(np.abs(z_scores['Date_Ordinal']) <= remove_outliers[1]) & (np.abs(z_scores['Intervals Total Length']) <= remove_outliers[1])]

    # Create a color map
    color_map = {
        1: "#1f77b4",
        2: "#ff7f0e",
        3: "#2ca02c",
        4: "#d62728",
        5: "#9467bd",
        6: "#8c564b",
        7: "#e377c2",
        8: "#7f7f7f",
        9: "#bcbd22",
        10: "#17becf"
    }

    # Fit a linear regression model
    X = df_copy[['Date_Ordinal']]
    y = df_copy[length_column]
    regression_model = LinearRegression()
    regression_model.fit(X, y)
    y_pred = regression_model.predict(X)

    # Create the figure
    plt.figure(figsize=figsize)

    # Plot the data points for each cluster
    for cluster in set(df_copy['Clusters']):
        plt.scatter(df_copy[df_copy['Clusters'] == cluster]['Collection_Date'],
                    df_copy[df_copy['Clusters'] == cluster][length_column],
                    color=color_map[cluster], label=f"Cluster: {cluster}", s=point_size)

    # Plot the regression line
    plt.plot(df_copy['Collection_Date'], y_pred, color='red', label='Regression Line')

    # Get the slope and intercept of the regression line
    slope = regression_model.coef_[0]
    intercept = regression_model.intercept_

    # Create a list of legend elements
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Cluster: {cluster}", markerfacecolor=color, markersize=10)
                       for cluster, color in color_map.items()]
    legend_elements.append(Line2D([0], [0], color='red', lw=2, label='Regression Line'))
    legend_elements.append(Line2D([0], [0], marker='None', color='w', label=f"Slope = {slope:.2f}\nIntercept = {intercept:.2f}"))

    # Customize the plot
    plt.legend(handles=legend_elements, loc=legend_loc)
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



def create_genbank_info_df_prog(intervals_file: str, show_progress: bool = False) -> pd.DataFrame:
    """
    This function fetches information for each GenBank ID from the NCBI database and stores it in a pandas DataFrame.

    Parameters:
    - intervals_file (str): The path to the file that contains the intervals. This can be the output of either the 'run_zhunt' or 'run_zdnabert' function.
    - show_progress (bool): Whether to show a progress bar while fetching information. Default is False.

    Returns:
    - df (pd.DataFrame): A pandas DataFrame containing the fetched information.
    """

    intervals = parse_prediction_files(intervals_file)
    ids =  list(intervals.keys())

    # Entrez email (required for accessing NCBI databases)
    Entrez.email = 'rustam_msu@mail.ru'

    def fetch_genbank_info(genbank_id: str) -> SeqIO.SeqRecord:
        handle = Entrez.efetch(db='nuccore', id=genbank_id, rettype='gb', retmode='text')
        record = SeqIO.read(handle, 'genbank')
        handle.close()
        return record

    def get_length_value(intervals: Dict[str, List[List[int]]], genbank_id: str) -> int:
        return sum(interval[1] - interval[0] for interval in intervals[genbank_id])

    def get_mean_length_value(intervals: Dict[str, List[List[int]]], genbank_id: str) -> float:
        lengths = [interval[1] - interval[0] for interval in intervals[genbank_id]]
        return sum(lengths) / len(lengths) if lengths else 0

    df = pd.DataFrame(columns=['GenBank ID', 'Accession', 'Description', 'Collection Date', 'Geographic Location',
                               'Sequence Length', 'Host', 'Intervals Total Length', 'Intervals Mean Length'])

    ids_iter = ids
    if show_progress:
        ids_iter = tqdm(ids, desc=f"Downloading GenBank info")
    for genbank_id in ids_iter:
        record = fetch_genbank_info(genbank_id)

        accession = record.id
        description = record.description

        collection_date = record.features[0].qualifiers.get('collection_date', '')
        if collection_date == '':
            collection_date = record.annotations.get('date', '')
        if isinstance(collection_date, list):
            collection_date = collection_date[0] if collection_date else ''
        geographic_location = record.features[0].qualifiers.get('country')

        features = record.features
        host = ''
        for feature in features:
            if feature.type == 'source' and 'host' in feature.qualifiers:
                host = feature.qualifiers['host'][0]
                break

        df = df.append({'GenBank ID': genbank_id,
                        'Accession': accession,
                        'Description': description,
                        'Collection Date': collection_date,
                        'Geographic Location': geographic_location,
                        'Sequence Length': len(record.seq),
                        'Host': host,
                        'Intervals Total Length': get_length_value(intervals, genbank_id),
                        'Intervals Mean Length': get_mean_length_value(intervals, genbank_id)
                        }, ignore_index=True)

    return df



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



def plot_zna_regions(regions: dict, data: List[Tuple[str, int, int]], colors: List[str], figsize: Tuple[int, int] = (30, 15)):
    """
    Plots the ZNA regions along with additional regions.

    Args:
        regions (dict): A dictionary of additional regions.
        data (List[Tuple[str, int, int]]): A list of tuples representing the ZNA regions.
        colors (List[str]): A list of colors corresponding to each ZNA region.
        figsize (Tuple[int, int]): The figure size. Defaults to (30, 15).
    """
    fig, ax = plt.subplots(figsize=figsize)
    genes, starts, ends = zip(*data)

    for i in range(len(data)):
        ax.hlines(y=0, xmin=starts[i], xmax=ends[i], linewidth=10, color=colors[i])

    # Set the x-axis limits to the extent of the genome
    lim = max(ends) + 50
    ax.set_xlim(0, lim)

    # Remove the y-axis ticks and labels
    ax.yaxis.set_visible(False)

    # Add a grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add padding to the bottom spine
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_position(('outward', 10))

    # Format the x-axis ticks and labels with gene names
    xticks = []
    xlabels = []
    for i in range(len(data)):
        xticks.append((starts[i] + ends[i]) / 2)
        xlabels.append(data[i][0])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=10)

    # Rotate the x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add a legend
    patches = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(data))]
    ax.legend(patches, genes, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5, fontsize=10)

    # Adjust the spacing between subplots to prevent overlap
    plt.subplots_adjust(bottom=0.35)

    # Plot additional regions
    counter = 1
    for key, value in regions.items():
        additional_data = [(region[0], region[1]) for region in value[1:]]
        additional_starts, additional_ends = zip(*additional_data)
        for i in range(len(additional_data)):
            ax.hlines(y=counter, xmin=additional_starts[i], xmax=additional_ends[i], linewidth=5)
        counter += 0.25

    # Add a title and axis labels
    title = list(regions.values())[0][0]
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Position (nucleotides)', fontsize=12)

    # Add the number of strains
    n_strains = len(regions)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f"Number of strains: {n_strains}", transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)



def zna_banch_prediction(file: str, function, function_params) -> None:
    """
    Perform a given function on sequences in a file and write the results to a new file.

    Args:
        file (str): The path to the input file.
        function (callable): The function to perform on the sequences.
        function_params (dict): The parameters to pass to the function.
    """
    with open(file, 'r') as input_file, open(f'{function.__name__}_{file}', "w") as output_file:
        file_name = input_file.name.split('/')[-1]
        output_file.write(f"{file_name}\n")

        for sequence_record in SeqIO.parse(input_file, format='fasta'):
            sequence_id = sequence_record.id
            output_file.write(f"{sequence_id}\n")
            sequence = str(sequence_record.seq)
            regions = function(sequence, **function_params)
            output_file.write("  start     end\n")

            for region in regions:
                output_file.write(f"   {region[0]}   {region[1]}\n")


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


def draw_boxplot_species(
        data_subset: pd.DataFrame,
        category_label: str = 'name',
        value_label: str = 'Intervals total length',
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
    species_groups = sorted([
        'Cats/Dogs/Swine', 'Bats', 'Fish', 'Birds', 'Whales',
        'Human/Cattle', 'Rodentia', 'Swine', 'Eulipotyphla', 'Human', 'Mink'
    ])

    # Filter the DataFrame to only include rows where 'species' is in species_groups
    data_subset = data_subset[data_subset['species'].isin(species_groups)]

    # Sort dataframe by species
    data_subset = data_subset.sort_values(by=['species', 'name'])

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
        hue='species',
        data=data_subset,
        dodge=False
    )

    # Add a title and labels
    ax.set_title(plot_title)
    ax.set_xlabel('Species')
    ax.set_ylabel(f'Z-RNA {value_label}')

    # Get color palette
    palette = sns.color_palette("husl", 11)

    # Create a mapping of labels to species
    label_to_species = data_subset.groupby(category_label)['species'].agg(pd.Series.mode).to_dict()

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
        taxa = data_subset[data_subset[category_label] == label_text.get_text()]['taxa'].unique()[0]
        # Convert first letter to Greek equivalent
        taxa = taxa.lower()
        taxa_greek = greek_dict.get(taxa, taxa[0])

        # Update y position to below y=0 and set color based on taxa
        ax.text(
            (i + 0.5) / len(categories), 0.01, taxa_greek,
            horizontalalignment='center', size='small', color='black',
            weight='semibold', transform=ax.transAxes,
            bbox=dict(facecolor=taxa_to_colors[taxa], alpha=0.5, boxstyle='round,pad=0.2')
        )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7, fontweight='bold')

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


def draw_simple_barplot(data: Dict, title: str = 'Count of variants', 
                 figsize: tuple = (15, 6), palette: str = "husl", 
                 desat: float = 0.6) -> None:
    """Draws a bar plot using seaborn and matplotlib.
    
    Args:
        data: A dictionary containing the data to be plotted. The keys represent the categories (x-values), and the 
              values represent the counts (y-values).
        title: A string representing the title of the plot. Default is 'Count of variants'.
        figsize: A tuple representing the size of the figure. Default is (15, 6).
        palette: A string representing the color palette to use. Default is 'husl'.
        desat: A float representing the desaturation level of the colors. Default is 0.6.
    """
    # Define color palette
    colors = sns.color_palette(palette, len(data), desat=desat)

    # Create a new figure
    plt.figure(figsize=figsize)

    # Plot the data
    bars = plt.bar(data.keys(), data.values(), color=colors, edgecolor='black')

    # Add labels and title
    plt.xlabel('Variant')
    plt.ylabel('Count')
    plt.title(title)

    # Show the plot
    plt.show()


# Data
data_6M = {
    'Alpha': 545082,
    'Beta': 7606,
    'Gamma': 27169,
    'Delta': 80825,
    'Epsilon': 41707,
    'Eta': 2683,
    'Iota': 33131,
    'Kappa': 563,
    'Lambda': 1328,
    'Mu': 5190,
    'Omicron': 2738343,
    'Zeta': 1309,
    'Unknown': 3060362
}


data_4k = {
    'Alpha': 304,
    'Beta': 5,
    'Gamma': 33,
    'Delta': 17,
    'Epsilon': 64,
    'Eta': 4,
    'Iota': 67,
    'Kappa': 1,
    'Lambda': 1,
    'Mu': 1,
    'Omicron': 2728,
    'Zeta': 1,
    'Unknown': 1097
}


def draw_boxplot_who(
        df_subset: pd.DataFrame,
        label: str = 'WHO label',
        values: str = 'Intervals total length',
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
    pangolin_values = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Eta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Omicron', 'Unknown', 'Zeta']

    # Create an empty list to store the 'Intervals total length' for each 'Pangolin' value
    data = []

    # For each 'Pangolin' value, append the 'Intervals total length' to the data list
    for pangolin in pangolin_values:
        data.append(df_subset[df_subset[label] == pangolin][values].values)

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


VIRUS_TAXA_SPECIES = {
    'Alphacoronavirus 1': ('Cats/Dogs/Swine', 'Alphacoronavirus'),
    'Alphacoronavirus AMALF': ('Bats', 'Alphacoronavirus'),
    'Alphacoronavirus BT020': ('Bats', 'Alphacoronavirus'),
    'Alphacoronavirus CHB25': ('Bats', 'Alphacoronavirus'),
    'Alphacoronavirus HKU33': ('Bats', 'Alphacoronavirus'),
    'Alphacoronavirus WA1087': ('Bats', 'Alphacoronavirus'),
    'Alphacoronavirus WA2028': ('Bats', 'Alphacoronavirus'),
    'Alphacoronavirus WA3607': ('Bats', 'Alphacoronavirus'),
    'Alphapironavirus bona': ('Fish', 'Alphapironavirus'),
    'Avian coronavirus': ('Birds', 'Gammacoronavirus'),
    'Avian coronavirus 9203': ('Birds', 'Gammacoronavirus'),
    'Bat Hp-betacoronavirus Zhejiang2013': ('Bats', 'Alphacoronavirus'),
    'Bat coronavirus CDPHE15': ('Bats', 'Alphacoronavirus'),
    'Bat coronavirus HKU10': ('Bats', 'Betacoronavirus'),
    'Beluga whale coronavirus SW1': ('Whales', 'Gammacoronavirus'),
    'Betacoronavirus 1': ('Human/Cattle', 'Betacoronavirus'),
    'Bulbul coronavirus HKU11': ('Birds', 'Deltacoronavirus'),
    'China Rattus coronavirus HKU24': ('Rodentia', 'Betacoronavirus'),
    'Common moorhen coronavirus HKU21': ('Birds', 'Deltacoronavirus'),
    'Coronavirus HKU15': ('Swine', 'Deltacoronavirus'),
    'Duck coronavirus 2714': ('Birds', 'Gammacoronavirus'),
    'Eidolon helvum bat coronavirus CMR704-P12': ('Bats', 'Betacoronavirus'),
    'Goose coronavirus CB17': ('Birds', 'Gammacoronavirus'),
    'Hedgehog coronavirus 1': ('Eulipotyphla', 'Betacoronavirus'),
    'Human coronavirus 229E': ('Human', 'Alphacoronavirus'),
    'Human coronavirus HKU1': ('Human', 'Betacoronavirus'),
    'Human coronavirus NL63': ('Human', 'Alphacoronavirus'),
    'Lucheng Rn rat coronavirus': ('Rodentia', 'Alphacoronavirus'),
    'MERS': ('Human', 'Betacoronavirus'),
    'Miniopterus bat coronavirus 1': ('Bats', 'Alphacoronavirus'),
    'Miniopterus bat coronavirus HKU8': ('Bats', 'Alphacoronavirus'),
    'Mink coronavirus 1': ('Mink', 'Alphacoronavirus'),
    'Munia coronavirus HKU13': ('Birds', 'Deltacoronavirus'),
    'Murine coronavirus': ('Rodentia', 'Betacoronavirus'),
    'Myodes coronavirus 2JL14': ('Rodentia', 'Betacoronavirus'),
    'Myotis ricketti alphacoronavirus Sax-2011': ('Bats', 'Alphacoronavirus'),
    'Night heron coronavirus HKU19': ('Birds', 'Deltacoronavirus'),
    'NL63-related bat coronavirus strain BtKYNL63-9b': ('Bats', 'Alphacoronavirus'),
    'Nyctalus velutinus alphacoronavirus SC-2013': ('Bats', 'Alphacoronavirus'),
    'Pipistrellus bat coronavirus HKU5': ('Bats', 'Betacoronavirus'),
    'Pipistrellus kuhlii coronavirus 3398': ('Bats', 'Alphacoronavirus'),
    'Porcine epidemic diarrhea virus': ('Swine', 'Alphacoronavirus'),
    'Rhinolophus bat coronavirus HKU2': ('Bats', 'Alphacoronavirus'),
    'Rhinolophus ferrumequinum alphacoronavirus HuB-2013': ('Bats', 'Alphacoronavirus'),
    'Rousettus bat coronavirus GCCDC1': ('Bats', 'Betacoronavirus'),
    'Rousettus bat coronavirus HKU9': ('Bats', 'Betacoronavirus'),
    'Scotophilus bat coronavirus 512': ('Bats', 'Alphacoronavirus'),
    'SARS': ('Human', 'Betacoronavirus'),
    'Sorex araneus coronavirus T14': ('Eulipotyphla', 'Alphacoronavirus'),
    'Suncus murinus coronavirus X74': ('Eulipotyphla', 'Alphacoronavirus'),
    'Tylonycteris bat coronavirus HKU4': ('Bats', 'Betacoronavirus'),
    'White-eye coronavirus HKU16': ('Birds', 'Deltacoronavirus'),
    'Wigeon coronavirus HKU20': ('Birds', 'Deltacoronavirus'),
    'SARS-CoV-2': ('Human', 'Betacoronavirus')
}
