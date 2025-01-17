import numpy as np
import matplotlib.pyplot as plt


def process_file(input_file):
    """
       processes the timestamp file generated from Linospad
       camera
    """
    line_count = 0
   
    with open(input_file, 'rb') as infile:
        i = 0
        array = np.fromfile(infile, dtype=np.uint32)
        results = np.zeros(len(array))
        for line in array:
            try:
                result = line -  (1 << 31)
                results[i] = result
                i += 1   
            except ValueError:
                print(f"Invalid number on line: {line}")
        print(results[0:10])
        print("done processing input file") 
        return results


def extract_pixel(input_array, pixel_num):
    print(f"Extracting Pixel {pixel_num}")
    total_pixels = 256
    lines_to_extract = 512
    skip_size = (lines_to_extract * total_pixels) - lines_to_extract
    start_point = lines_to_extract * pixel_num
    result_array = np.ones(len(input_array)//200)*(-999)  # Initialize an empty NumPy array to store the chunks

    current_position = start_point
    i = 0

    while current_position < input_array.size:
        # Get the current chunk of elements (first N)
        current_chunk = input_array[current_position:current_position + lines_to_extract]

        # Append the current chunk to the result array
        #result_array = np.concatenate((result_array, current_chunk))
        
        result_array[i: i + lines_to_extract] = current_chunk 
        
        i += lines_to_extract
        
        # Move to the next chunk (skip the next N elements)
        current_position += lines_to_extract + skip_size
        
    result_array = result_array[0 : i]
    print(np.all(result_array[i:] == -999)) 
    #np.savetxt(output_pixel_name, result_array, delimiter=' ', newline='\n', header='', footer='', encoding=None)
    
    return result_array
    
def create_counts(data):
    """
      takes data from clean timestamps,
      bins the data and computes the calibration
      matrix for the TDC
    """
    #all data should be a value between 0-139
    data = data[data >= 0]
    data = data % 140    #create bin boundaries from 0 to 141
    bins = np.arange(0, 141, 1)
    #create a histogram of the data, use the counts to calculate the bin width
    counts, bin_edges = np.histogram(data, bins=bins)
    
    return counts

def create_counts_big_M(data):
    """
      takes data from clean timestamps,
      bins the data and computes the calibration
      matrix for the TDC
    """
    #all data should be a value between 0-139
    data = data[data >= 0]   #create bin boundaries from 0 to 141
    bins = np.arange(0, 2801, 1)
    #create a histogram of the data, use the counts to calculate the bin width
    counts, bin_edges = np.histogram(data, bins=bins)
    
    return counts


def remove_empty_bins(numbers):
    
    # Sample list of numbers in random order
    numbers = [5, 2, 8, 1, 7, 2, 7, 2, 8]

   # Create a sorted array of unique numbers
    sorted_unique_numbers = np.unique(numbers)
    sorted_unique_numbers.sort()

    # Create a mapping of original numbers to renumbered values
    number_mapping = {num: i + 1 for i, num in enumerate(sorted_unique_numbers)}

    # Create a new array with the numbers renumbered based on their value
    renumbered_numbers = np.array([number_mapping[num] for num in numbers])
    return renumbered_numbers


def fraction_overlap(top, bottom, pixel_num):
    '''A1 = top left
       A2 = top right 
       B1 = Bottom Left
       B2 = Bottom right 
    '''
    A1 = top[0]
    A2 = top[1]
    B1 = bottom[0]
    B2 = bottom[1]
    if (A2 - A1) == 0: 
        return 0
     
    fraction_overlap_internal = max(0, min(B2, A2) - max(B1, A1))/min(A2 - A1, B2 - B1)
  
    if min(A2 - A1, B2 - B1) == B2 - B1:
        print(f"Warning: input on {pixel_num} is larger than output bin!")
    return fraction_overlap_internal
   

def calc_dnl(input_counts):
    """
    calculate the DNL for a given distribution
    input_counts = the target distribution to calculate counts
    """
    num_bins = len(input_counts[np.nonzero(input_counts)])
    lsb_ideal = np.cumsum(input_counts).max()/num_bins
    #print(LSB)
    output_array = np.zeros(num_bins)
    #print(num_bins)
    clean_counts = input_counts[np.nonzero(input_counts)]
    #print(counts)
    i = 0
    for count in clean_counts:
        output_array[i] = count/lsb_ideal - 1
        i = i + 1
    return output_array
    
def create_calibration_matrix(data, output_matrix_name, pixel_num, num_output_bins):
    """
      takes data from clean timestamps,
      bins the data and computes the calibration
      matrix for the TDC
    """
    print(f"creating calibration matrix for {pixel_num}")
    #all data should be a value between 0-139
    data = data[data >= 0]
    print(f"{data[0:3]=}")
    data = data % 140
    #create bin boundaries from 0 to 141
    hist_input_bins = np.arange(0, 141, 1)
    #create a histogram of the data, use the counts to calculate the bin width
    counts_1, _ = np.histogram(data, bins= hist_input_bins)
    #calculate x bin width 
    x = np.cumsum(counts_1) / np.cumsum(counts_1).max() * 140
    #cut off here for linearization curve
    x_edges = np.concatenate(([0], x))
    y_edges = np.linspace(0, 141, num_output_bins + 1)
    M = np.zeros((len(x_edges)-1, len(y_edges)-1))
    for i in range(len(x_edges)-1): 
        for j in range(len(y_edges)-1):
            M[i, j] = fraction_overlap((x_edges[i], x_edges[i+1]), (y_edges[j], y_edges[j+1]), pixel_num)
    np.savetxt(output_matrix_name, M.T, delimiter=' ', newline='\n', header='', footer='', encoding=None)
    return M.T, counts_1

def load_matrix(output_matrix_name):
    '''loads the matrix from for the user specified pixel
        returns the matrix from the fxn
    '''
    matrix = np.genfromtxt(output_matrix_name)

    return matrix

def create_all_calibration_matricies(cleaned_calibration_data, num_output_bins):
    '''
    fxn creates calibration matricies for all 64 starting pixels
    cleaned_calibration_data = data from process_file fxn
    num_output_bins = a constant that determines how much the distribution will be undersamples
    '''
    print("extracting pixels and creating calibration matricies")
    for i in range(64):
        pixel_num = i
        output_matrix_name = f"pixel_{pixel_num}_calibration_matrix.txt"
        ext_pixel_data = extract_pixel(cleaned_calibration_data, pixel_num)
        create_calibration_matrix(ext_pixel_data, output_matrix_name, pixel_num, num_output_bins)
    print("done extracting pixel")

def chart(input_pixel_data, output_bin_counts, bin_array, num_output_bins, Y_LIM):
    '''
    creates the before and after chart for the specified pixel
    input_pixel_data = extracted pixel data
    output_bin_counts = calculated bin counts for calibrated data
    bin_array = bin boundary for first chart
    '''
    # An "interface" to matplotlib.axes.Axes.hist() method
    _ , ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5, 5))

    # chart original data
    n, _ , _ = ax1.hist(input_pixel_data, bins = bin_array, color='blue', alpha=0.7)
    ax1.set_title('Step 1: Pre-Calibration/Linearization')
    ax1.set_xlabel('Bins (0-140)')
    ax1.set_ylabel('Number of Photons Recieved')
    ax1.set_ylim(top= Y_LIM)
    dnl_1 = calc_dnl(n)

    # chart calibrated data
    ax2.bar(np.arange(0, num_output_bins, 1), output_bin_counts, width = 0.8, align='center', color='green')
    ax2.set_title('Step 2: Post-Calibration/Linearization')
    ax2.set_xlabel('Bins (0-50)')
    ax2.set_ylabel('Number of Photons Recieved')
    ax2.set_ylim(top= Y_LIM)
    dnl_2 = calc_dnl(output_bin_counts)

    # chart original DNL
    ax3.bar(range(len(dnl_1)), dnl_1, color ='maroon', width = 0.4)
    ax3.set_title('DNL Pre-Calibration')
    ax3.set_xlabel(f'DNL {max(dnl_1)=:.2f} {min(dnl_1)=:.2f}')
    ax3.set_ylabel('Value')

    # chart calibrated DNL
    ax4.bar(range(len(dnl_2)), dnl_2, color ='red', width = 0.4)
    ax4.set_title('DNL Post-Calibration/Linearization')
    ax4.set_xlabel(f'DNL {max(dnl_2)=:.2f} {min(dnl_2)=:.2f}')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()
    
    

if __name__ == '__main__': 

    CALIBRATION_DATA_FILE = r"/disk/nobackup/galante/11_2_23_2_timestamps_laser_10ms_600_cycles0000000001.dat"
    #board_name='NL28'
    PIXEL_FOR_RUN = 0
    MATRIX_FOR_RUN = PIXEL_FOR_RUN % 64
    LOAD_MATRIX_NAME = f"pixel_{MATRIX_FOR_RUN}_calibration_matrix.txt"
    NUM_OUTPUT_BINS = 40
    NUM_OUTPUT_BINS_BIG_M = 800
    YLIM = 50_000
    
    #for calibrating all pixels
    cleaned_matrix = process_file(CALIBRATION_DATA_FILE)
    #create_all_calibration_matricies(cleaned_matrix, NUM_OUTPUT_BINS)
    
    #for running on 1 pixel
    pixel_data = extract_pixel(cleaned_matrix, PIXEL_FOR_RUN)
    M = load_matrix(LOAD_MATRIX_NAME)
    counts = create_counts(pixel_data)
    print(f"{np.sum(counts)=}")
    print(f"{np.shape(counts)=}")

    output_counts = np.dot(M, counts)
    print(f"{np.sum(output_counts)=}")
    bins = np.arange(0, 141, 1)
    print("Saving txt file")

    #big chart stuff
    counts2 = create_counts_big_M(pixel_data)
    print(f"{np.shape(counts2)=}")
    big_M = np.kron(np.eye(20, dtype = int), M)
    print(np.shape(big_M))
    bins2 = np.arange(0, 2801, 1)
    output_counts2 = np.dot(big_M, counts2)


    #make both charts?
    #chart(pixel_data, output_counts, bins, NUM_OUTPUT_BINS, YLIM)
    chart(pixel_data, output_counts2, bins2, NUM_OUTPUT_BINS_BIG_M, max(output_counts2)+3000)

    #try plotting with semi-log y-axis