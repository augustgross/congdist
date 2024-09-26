import pandas as pd
import os
import numpy as np
from scipy.spatial.distance import euclidean

# List of states and file categories
states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 
          'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 
          'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
categories = ['02-SOCIAL', '03-ECONOMIC', '04-HOUSING', '05-DEMOGRAPHICS']

# New root directory for filtered results
output_root_dir = 'results'

# Function to process each file
def process_file(file_path, category, geoname_vectors):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert 'PCT_ESTIMATE' to numeric, forcing errors to NaN
    df['PCT_ESTIMATE'] = pd.to_numeric(df['PCT_ESTIMATE'], errors='coerce')

    # Filter rows where PCT_ESTIMATE is less than or equal to 100
    filtered_df = df[df['PCT_ESTIMATE'] <= 100]

    # Group by 'GEONAME' and append the vector for each
    for geoname, group in filtered_df.groupby('GEONAME'):
        # Collect PCT_ESTIMATE values into a list
        vector = group['PCT_ESTIMATE'].tolist()

        # If this GEONAME exists in other categories, concatenate the vector
        if geoname in geoname_vectors:
            geoname_vectors[geoname][category] = vector
        else:
            geoname_vectors[geoname] = {category: vector}

# Dictionary to hold the vectors for each GEONAME across all categories and states
geoname_vectors = {}

# Main loop to process all state files and categories
for state in states:
    for category in categories:
        # Construct the file path for each state and category
        file_path = f'{category}/ALLCD_DP{category.split("-")[0]}_{state}.csv'
        
        # Process each file if it exists
        if os.path.exists(file_path):
            process_file(file_path, category, geoname_vectors)
        else:
            print(f"File not found: {file_path}")

def get_geoname_vectors(target_geoname):
    if target_geoname in geoname_vectors:
        target_vectors = geoname_vectors[target_geoname]
    else:
        print(f"Target GEONAME {target_geoname} not found.")
        target_vectors = None

    # Compute Euclidean distances for each category between the target vector and all other GEONAMES
    distances_by_category = {category: [] for category in categories}
    combined_distances = []

    if target_vectors:
        for geoname, vectors in geoname_vectors.items():
            if geoname != target_geoname:
                combined_vector_target = []
                combined_vector_other = []
                for category in categories:
                    # Ensure the target vector and comparison vector exist and are of the same length (truncate if needed)
                    if category in target_vectors and category in vectors:
                        target_vector = target_vectors[category]
                        comparison_vector = vectors[category]
                        
                        min_length = min(len(target_vector), len(comparison_vector))
                        target_vec_truncated = target_vector[:min_length]
                        comparison_vec_truncated = comparison_vector[:min_length]
                        
                        # Concatenate the vectors for the combined category
                        combined_vector_target.extend(target_vec_truncated)
                        combined_vector_other.extend(comparison_vec_truncated)
                        
                        # Calculate the Euclidean distance for this category
                        distance = euclidean(target_vec_truncated, comparison_vec_truncated)
                        distances_by_category[category].append((geoname, distance))

                # Calculate the Euclidean distance for the combined vector
                if combined_vector_target and combined_vector_other:
                    combined_distance = euclidean(combined_vector_target, combined_vector_other)
                    combined_distances.append((geoname, combined_distance))

    pd.set_option('display.max_colwidth', None)  # No truncation of column content


    # Convert distances to DataFrames for better readability and sorting
    for category, distances in distances_by_category.items():
        distances_df = pd.DataFrame(distances, columns=['GEONAME', 'Distance'])
        distances_df = distances_df.sort_values(by='Distance')

        # Save the distances to a CSV file for each category
        distances_output_file = os.path.join(output_root_dir, f'euclidean_distances_{category}.csv')
        distances_df.to_csv(distances_output_file, index=False)

        # Display the closest districts for the category (optional)
        print(f"\nClosest districts for category {category}:")
        print(distances_df.head())

    # Process the combined category distances
    combined_distances_df = pd.DataFrame(combined_distances, columns=['GEONAME', 'Distance'])
    combined_distances_df = combined_distances_df.sort_values(by='Distance')

    # Save the combined distances to a CSV file
    combined_output_file = os.path.join(output_root_dir, 'euclidean_distances_combined.csv')
    combined_distances_df.to_csv(combined_output_file, index=False)

    # Display the closest districts for the combined category (optional)
    print("\nClosest districts for combined categories:")
    print(combined_distances_df.head())

if __name__ == "__main__":
    print("Enter a state (e.g. California, New York, etc.)")
    state = input()
    print("Enter a district (e.g. 1, (at Large), 2, etc.)")
    district = input()
    target_geoname = f"Congressional District {district} (118th Congress), {state}"
    get_geoname_vectors(target_geoname)